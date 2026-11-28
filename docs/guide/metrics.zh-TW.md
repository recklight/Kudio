# 品質評估 {#measuring-quality}

這一頁的每個指標，都是拿處理過的錄音和它原本那段乾淨錄音比較來評分。
沒有乾淨錄音時，[檢查錄音](inspection.md)裡的檢查還是能用，DNSMOS 也在其中；
這一頁的東西則全都需要參考訊號。

其中 SNR、SI-SDR、分段 SNR 和 SDI 這四個，還會拿一個訊號的第 *i* 個樣本
和另一個訊號的第 *i* 個樣本比。只要延遲一個樣本，完美複本的 SI-SDR 就會掉到
+11.9 dB，只比這一頁的降噪器在完全沒有延遲時的分數（+11.7 dB）高一點，
而分數裡沒有任何地方告訴你出問題的是延遲。SDR 容得下較晚開始，STOI 容得下
大約一毫秒，PESQ 會自己把兩者對齊，但它們也都不會回報延遲。所以要先對齊，
再挑指標。

## 先算一個分數 {#a-first-score}

這一頁的範例共用一段合成片段：兩段由諧波音組成的語句疊在環境音上，中間隔著一段停頓，
用 [`add_noise_snr`][kudio.effects.augment.add_noise_snr] 以 5 dB SNR 混入白雜訊，
再用 [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] 降噪
（見[降噪](enhancement.md)）。所有亂數都給了種子，所以除了 100 dB 以上那些
屬於捨入誤差的數字，下面的數字在你的電腦上應該都會一樣。

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
```

[`score`][kudio.core.evaluator.score] 的第一個參數是參考訊號，第二個是要評的訊號，
它依名稱執行指標，回傳一個 dict。不給清單時，它會跑 `snr`、`si_sdr`、`segsnr`、
`sdi`，以及 `pesq`、`stoi`、`estoi`、`sdr` 之中這台電腦在這個取樣率下能跑的那些，在這裡就是
PESQ 以外的全部。這一頁後面從 [Hu 與 Loizou 的指標](#hu-and-loizous-measures)
開始的那些，要指名才會跑：

```python
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

`snr`、`si_sdr`、`segsnr` 和 `sdi` 純粹用 numpy 計算，除了 kudio 什麼都不用裝。
它們也各自是獨立的函式，參數順序相同：[`snr`][kudio.core.evaluator.snr]、
[`si_sdr`][kudio.core.evaluator.si_sdr]、
[`segmental_snr`][kudio.core.evaluator.segmental_snr] 和
[`sdi`][kudio.core.evaluator.sdi]，以及 `stoi`、`estoi` 和 `sdr`；
`pesq` 需要 `kudio[pesq]`，而且取樣率必須是 8 或 16 kHz。指名一個跑不了的指標時，
`score` 會引發例外，而不是把它略過：未知的名稱是 `ValueError`，缺少套件則是
[`DependencyError`][kudio.exceptions.DependencyError]，訊息裡會寫出要安裝的選用套件組。

[`check_metrics_install`][kudio.core.evaluator.check_metrics_install] 回傳
`(pesq_ok, stoi_ok, sdr_ok)`，也就是這台電腦有哪些後端；每個缺少的後端，
它在每個行程裡只記錄一次安裝提示。`AudioEvaluate` 和 `eval_metrics` 也會呼叫它，
所以請在第一次呼叫它們之前設定好 logging，不然那唯一一次的提示就沒人看得到了。

[`eval_metrics`][kudio.core.evaluator.eval_metrics] 是固定形狀的版本，給要拆成
三個值的程式碼用：一律回傳 `(pesq, stoi, sdr)`，沒有安裝、無法在這一對上執行，
或執行失敗的，就是 `None`。和這一頁其他所有東西不同，它的取樣率放在**第一個**：
`kudio.eval_metrics(sr, clean, noisy)` 在這裡回傳
`(None, 0.6065905900244255, 5.010295956402773)`。

## 先把兩者對齊 {#line-the-pair-up-first}

下面是這段片段和自己比的分數，以及 log-MMSE 的輸出和這段片段比的分數，
兩者都晚了幾個樣本才開始。兩者的長度都沒變，所以長度檢查沒有東西可以察覺：

![SI-SDR 和延遲的關係：晚一個樣本的完美複本得到 +11.9 dB，從晚四個樣本開始，複本和 log-MMSE 的輸出都低於未處理的含雜訊輸入](../images/metrics-delay-light.webp#only-light)
![SI-SDR 和延遲的關係：晚一個樣本的完美複本得到 +11.9 dB，從晚四個樣本開始，複本和 log-MMSE 的輸出都低於未處理的含雜訊輸入](../images/metrics-delay-dark.webp#only-dark)

晚四個樣本，在 16 kHz 是四分之一毫秒，這時複本得到 +3.9 dB，降噪後的版本得到
+4.1 dB，兩者都比完全沒處理過的含雜訊輸入（+5.0 dB）還低。從三個樣本開始，
兩條曲線的差距都在 0.4 dB 以內：分數由延遲決定，降噪的效果幾乎看不出來。
不知道有延遲的人讀到這個數字，會以為降噪器比什麼都不做還糟。

延遲的來源都很平常：因果濾波器、重新取樣器、會對輸入音框做 padding 的模型、
用喇叭播出來再錄一次的錄音。處理過的檔案比乾淨的長時，
[長度檢查](#when-the-lengths-differ)會發出警告；如果有某個步驟把它裁回輸入的長度，
就什麼警告都沒有。

### 找出延遲 {#finding-the-delay}

[`find_delay`][kudio.core.align.find_delay] 用互相關量出延遲，精確到最接近的樣本，
[`align`][kudio.core.align.align] 則回傳裁到共同部分的這一對：

```python
processed = np.concatenate([np.zeros(128, np.float32), enhanced])  # 8 ms late

found = kudio.find_delay(clean, processed, sr)
print(found)

a, b = kudio.align(clean, processed, sr)
print(len(a), len(b), round(kudio.si_sdr(a, b), 2))
```

```text
lags by 128 samples (8.0 ms), correlation 0.968
64000 64000 11.72
```

同一對不對齊直接評分，會得到 -7.97 dB，外加一則長度不同的警告。

`find_delay` 回傳的 [`Alignment`][kudio.core.align.Alignment] 裡，`delay` 為正
代表第二個訊號比較晚開始。相關係數只在重疊的部分上正規化，所以一段長檔案裡
只共享一秒的真正配對，分數還是會很高。延遲固定的處理鏈只需要在大聲的段落量一次，
之後的每一對都可以用 [`Alignment.apply`][kudio.core.align.Alignment.apply]
照已經找到的延遲裁切：

```python
other = kudio.add_noise_snr(clean, noise, snr_db=10, seed=0)
other_out = np.concatenate([np.zeros(128, np.float32),
                            kudio.spectral_enhance(other, sr)])

a, b = found.apply(clean, other_out)     # the 128 samples measured above
print(round(kudio.si_sdr(a, b), 2))
```

```text
14.74
```

### 拿錯檔案也會得到延遲 {#the-wrong-file-still-gets-a-delay}

互相關一定有最大值，所以 `find_delay` 對任何兩個訊號都會回傳一個延遲，
包括毫不相干的兩個。能區分這兩種情況的是相關係數：低於
[`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION]，也就是 0.5 時，
`align` 會拒絕：

```python
print(kudio.find_delay(clean, noise, sr))
kudio.align(clean, noise, sr)
```

```text
leads by 5873 samples (367.1 ms), correlation 0.016, polarity inverted
kudio.exceptions.FeatureError: these do not look like the same recording: best correlation 0.016 at -5873 samples, below 0.5. Aligning them would put a number on two unrelated signals; pass min_correlation=0 to do it anyway.
```

相關係數只有 0.016 時，那個極性的註記同樣沒有意義。

門檻是我刻意設低的。上面降噪後的版本是 0.968；這個門檻是用來抓出拿錯檔案的情況，
不是用來替處理打分數。

通過門檻，也不代表那個延遲就是你要的。語音會重複，在長檔案上不設範圍地搜尋，
可能會在半分鐘外找到更好的配對。知道延遲很小時，用 `max_seconds=` 限制搜尋範圍，
設在你預期的最大延遲之上。回傳的是範圍內最好的延遲，就算真正的延遲在範圍外，
它還是可能通過 0.5。

### 極性 {#polarity}

極性相反的複本，相關係數是 -1。它是同一段錄音，只是處理鏈中某個環節把它翻過來了。
我把這當成一項發現，而不是順手修正掉的東西：`find_delay` 會回報
`polarity inverted`，`align` 則記錄一則警告（設定好 logging 才看得到），
並原樣回傳這一對。SI-SDR 看不出翻轉，SNR 看得出來：和 `-processed` 對齊後，
上面那一對的 SI-SDR 仍然是 11.72 dB，SNR 則掉到 -5.77 dB。

### 單聲道，同一個取樣率 {#mono-at-one-rate}

`find_delay` 和 `align` 只接受單聲道。立體聲陣列會被拒絕，而不是混成單聲道，
因為哪個聲道領先本身就是另一個問題。指標也會以 `ValueError` 拒絕立體聲，
不會把兩個聲道當成一個訊號來評分；請一次評一個聲道，或先混成單聲道。
一個聲道可以是向量、一欄 `(n, 1)` 或一列 `(1, n)`。
兩個訊號也都必須是你傳入的那一個 `sr`。不是的時候 `find_delay` 分辨不出來，
所以請先用 [`resample`][kudio.core.io.resample] 把它們調成同一個取樣率。

只有一對檔案時，`kudio align clean.wav processed.wav --metrics` 會印出延遲，
以及對齊前後的 SI-SDR，兩個檔案看起來不像同一段錄音時會以 1 結束
（見[命令列](../cli.md)）。

## 長度不同時 {#when-the-lengths-differ}

[`snr`][kudio.core.evaluator.snr]、`si_sdr`、`segmental_snr` 和 `sdi`
會把兩個訊號裁到共同的長度，`score` 和 `eval_metrics` 對它們執行的指標，除了 PESQ
以外也都一樣。
較長那個的尾端永遠不會被比較到，所以只要長度不同，每次呼叫都會發出警告：

```python
kudio.snr(clean, clean[:60000])
```

```text
UserWarning: snr(): ref is 64000 samples and deg 60000: only the first 60000 samples are scored and the rest is ignored. Use kudio.align if the short one is delayed rather than just short, or trim the pair yourself to say which part you meant.
```

同一行程式碼發出的相同警告，Python 只會顯示一次，所以在迴圈裡處理一堆
都被裁掉同樣長度的配對時，只有第一對會警告；`warnings.simplefilter("always")`
會讓每一次都顯示出來。

當較長那個訊號有超過一半會量不到時，這一對就會被拒絕。沒有這個上限的話，
片段的第一秒會被評成整整四秒的完美複本，因為它缺少的那三秒根本不會被看到：

```python
kudio.snr(clean, clean[:16000])
```

```text
ValueError: snr(): ref is 64000 samples and deg 16000: comparing them leaves 48000 of the longer one's 64000 samples unmeasured (75.0%), past the 50% limit, so the score would be about the overlap rather than about the pair. Check that these two are the same utterance, line them up with kudio.align, or raise kudio.core.evaluator.MAX_LENGTH_MISMATCH to score them as they sit.
```

這個上限是 [`MAX_LENGTH_MISMATCH`][kudio.core.evaluator.MAX_LENGTH_MISMATCH]，
也就是較長訊號的 0.5。真實的處理流程在尾端會少掉一兩個視窗，或因為重新取樣器的
捨入少掉一個樣本，半個檔案遠遠超過這種程度。每次呼叫都會重新讀取這個常數，
所以長度真的差這麼多的一對，可以在設定
`kudio.core.evaluator.MAX_LENGTH_MISMATCH = 0.8` 之後評分。它必須設在那個模組上：
並沒有 `kudio.MAX_LENGTH_MISMATCH` 這個名稱，指定一個也不會改變任何東西。

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] 會用同樣的方式檢查它讀入的
每一對檔案，訊息裡會寫出兩個檔名。拒絕時引發的是普通的 `ValueError`，所以在批次外面
包一層攔 [`KudioError`][kudio.exceptions.KudioError] 的 `except`，是攔不到它的。

`score`、`eval_metrics` 和 `AudioEvaluate` 執行的指標裡，只有 PESQ 拿到完整的兩段：
[`pesq`][kudio.core.evaluator.pesq]，以及這三者和 `composite` 算的 PESQ，都照原樣
接收兩個訊號，跟 ITU 程式讀兩個檔案一樣。PESQ 會自己找出兩者之間的延遲，並且在兩段
完整的長度上計算，所以就算裁掉的只是靜音，分數也會變：在 ITU 一致性測試 2(b) 的
40 對上，較短檔案之後的尾巴都是靜音或低位準雜訊，先裁再算仍讓分數最多差了 0.56
（[指標的驗證方式](../validation.md#pesq)）。PESQ 和會裁切的指標一起跑時，長度警告會
註明 PESQ 拿到的是完整的兩段；只要求 PESQ 的 `score`，以及 `AudioEvaluate.pesq_score`，
完全不檢查長度。

## 每個指標看到什麼 {#what-each-metric-sees}

同一段片段，以六種方式劣化，再用六個指標評分：

| 劣化方式 | `snr` | `si_sdr` | `segsnr` | `sdi` | `stoi` | `sdr` |
|---|---:|---:|---:|---:|---:|---:|
| 雜訊，5 dB SNR | 5.0 | 5.0 | -2.1 | 0.32 | 0.61 | 5.0 |
| 增益減半 | 6.0 | 138.9 | 6.0 | 0.25 | 1.00 | 283.0 |
| 極性翻轉 | -6.0 | 144.9 | -6.0 | 4.00 | 1.00 | 283.0 |
| 低通，2 kHz | 18.2 | 18.2 | 10.3 | 0.02 | 0.93 | 24.9 |
| 晚 1 個樣本 | 12.1 | 11.9 | 4.6 | 0.06 | 1.00 | 91.3 |
| 晚 1 ms | -1.4 | -10.0 | -2.2 | 1.40 | 0.99 | 53.6 |

??? note "產生這張表的腳本"

    ```python
    def late(y, n):
        return np.concatenate([np.zeros(n, np.float32), y])[:len(y)]

    cases = {
        "noise, 5 dB SNR": noisy,
        "half the gain": 0.5 * clean,
        "polarity flipped": -clean,
        "low-pass, 2 kHz": kudio.lowpass(clean, sr, cutoff=2000),
        "1 sample late": late(clean, 1),
        "1 ms late": late(clean, 16),
    }
    names = ["snr", "si_sdr", "segsnr", "sdi", "stoi", "sdr"]
    for case, deg in cases.items():
        s = kudio.score(clean, deg, sr, names)
        print(f"| {case} | {s['snr']:.1f} | {s['si_sdr']:.1f} | {s['segsnr']:.1f} "
              f"| {s['sdi']:.2f} | {s['stoi']:.2f} | {s['sdr']:.1f} |")
    ```

超過 100 dB 的數字，代表指標沒有發現任何問題。在 `snr` 和 `si_sdr` 裡，
這個數字來自防止除以零的 1e-12，所以會隨片段的能量變動，138.9 和 144.9
是同一個判定。`sdr` 的 283.0 則是 mir_eval 殘差裡留下的捨入誤差。

[`snr`][kudio.core.evaluator.snr]
:   每一種差異都會扣分：雜訊、增益、極性、濾波、延遲。音量只有一半的完美複本
    得到 6.0 dB。音量本身也是你要評判的一部分時，就用它。

[`si_sdr`][kudio.core.evaluator.si_sdr]
:   比較之前，先去掉兩個訊號的平均值，再去掉最吻合的增益[^sisdr]，所以音量改變
    或極性翻轉不會扣分，濾波和延遲則仍然會。評估降噪器時，我第一個看的就是它，
    因為降噪器不該因為輸出變小聲而被扣分。

`segsnr`、[`segmental_snr`][kudio.core.evaluator.segmental_snr]
:   每個音框各算一次 SNR，限制在 [-10, 35] dB 之間，再取平均。透過 `score` 時，
    音框一律是 512 個樣本、位移 256，在 16 kHz 是 32 ms，在 48 kHz 是 10.7 ms；
    要改的話，`segmental_snr` 接受 `frame_length=` 和 `hop_length=`。
    停頓和語音算得一樣重：在 5 dB 的混音裡，249 個音框中有 116 個落在
    -10 dB 的下限，大多是語句周圍的環境音，這把平均拉到 -2.1 dB。
    完美複本剛好得到 35.0，前提是它沒有任何音框是數位靜音：這樣的音框得 -10 dB，
    輸出在那裡也是靜音時得 0 dB，並有一則警告算出有幾個。比一個音框還短的片段
    得到 nan，也會有警告。

[`sdi`][kudio.core.evaluator.sdi]
:   `sum((ref - deg)**2) / sum(ref**2)`，也就是線性尺度的 SNR，
    `10 ** (-snr / 10)`：0 是完美複本，1 代表誤差和語音一樣多，越低越好。
    它是對整個輸出計算的，所以殘留雜訊和失真一樣算成誤差，未處理的 5 dB 混音
    得到 0.32。降噪文獻裡的語音失真指數[^sdi]，是讓乾淨語音單獨通過濾波器，
    把雜訊排除在外，未經處理的輸入在那裡得到 0。`sdi` 只能和用同樣方式算出的
    數字比較。

`stoi`、`estoi`
:   短時客觀可懂度（short-time objective intelligibility）[^stoi]，數字和 pystoi
    相同。它先把兩個訊號重新取樣到 10 kHz、去掉靜音音框，再比較三分之一八度頻帶
    裡的短時包絡，所以增益、極性和一毫秒的延遲幾乎動不了它，雜訊則會。
    它能容忍的也差不多就這些：晚 5 ms 得到 0.85，晚 10 ms 得到 0.63，
    大約就是 5 dB 雜訊的代價（0.61）。擴充版的 `estoi` 在 5 dB 混音上得到 0.31，
    STOI 則是 0.61。完美複本兩者都得到 1，越高越好。pystoi 在去掉靜音音框之後，
    需要剩下大約 0.4 秒的音訊；不到的話，就算是完美複本，它也會回傳 0.00001，
    這是佔位值而不是分數，所以 kudio 會把這種片段評為 NaN 並發出警告，
    `AudioEvaluate` 則把它排除在平均之外。

`sdr`
:   BSS Eval 的 SDR[^bss]，數字和 mir_eval 逐位元相同。它從參考訊號到輸出擬合出最佳的
    512 tap 濾波器（512 是 mir_eval 的長度，論文本身沒有固定），只把這個濾波器產生不出來的部分算成失真。增益、極性、低通，
    以及最多 511 個樣本的延後開始，都會部分或完全被容忍，所以晚一個樣本得到
    91.3 dB。這個濾波器是因果的，提早開始的輸出完全享受不到這些：早一個樣本
    得到 15.2 dB。只拿它和用同樣方式算出的 SDR 數字比較，絕對不要和 SI-SDR 比。

不同的指標，對同一個降噪器的看法也會不一致。在這段片段上，log-MMSE 讓 SI-SDR
增加 6.8 dB（從 4.97 到 11.72），STOI 卻停在原地（從 0.607 到 0.601），
ESTOI 甚至下降了，從 0.305 降到 0.276。

## PESQ：講清楚是哪一種 {#pesq-say-which-one}

PESQ 有兩個標準、兩種尺度，同一個檔案會得到不同的數字。[^p862] ITU-T P.862 評的是
窄頻語音，kudio 在 8 或 16 kHz 跑它；P.862.2 是它的寬頻延伸。P.862.2 給了 8 和
16 kHz 兩組輸入濾波器，但 P.862 的 Annex A 規定寬頻的實作要在 16 kHz 下執行，
參考程式、`pesq` 套件和 kudio 都照這樣做。P.862 可以
回報原始分數，範圍是 -0.5 到 4.5，也可以經由 P.862.1 對應到 MOS-LQO。P.862.2
只回報 MOS-LQO：它的原始模型輸出一定會經過它自己的對應式。
[`pesq`][kudio.core.evaluator.pesq] 用 `mode=` 指定頻寬、用 `scale=` 指定尺度，
而不是不聲不響地替你挑一個；`score`、`eval_metrics` 和 `AudioEvaluate`
則以 `pesq_mode=` 和 `pesq_scale=` 把它們傳下去：

| `mode` | `scale` | 8 kHz 時 | 16 kHz 時 |
|---|---|---|---|
| `'auto'`（預設） | `'lqo'`（預設） | P.862，MOS-LQO | P.862.2，MOS-LQO |
| `'nb'` | `'lqo'` | P.862，MOS-LQO | P.862，MOS-LQO |
| `'wb'` | `'lqo'` | 拒絕 | P.862.2，MOS-LQO |
| `'auto'` 或 `'nb'` | `'raw'` | P.862，原始分數 | P.862，原始分數 |
| `'wb'` | `'raw'` | 拒絕 | 拒絕 |

其他取樣率會引發 `ValueError`，訊息會叫你先重新取樣。

論文裡只寫「PESQ」的，可能是原始的 P.862 分數，也可能是 MOS-LQO，可能是窄頻，
也可能是寬頻，所以把你的數字和它並列之前，先弄清楚是哪一種。數字要放進表格時，
我會兩個參數都傳，並在旁邊註明尺度。這兩行都需要 `pesq` 套件，它要從原始碼編譯
（需要什麼，請看[入門](../getting-started.md)）：

```python
kudio.pesq(clean, enhanced, 16000)                          # P.862.2 MOS-LQO
kudio.pesq(clean, enhanced, 16000, mode="nb", scale="raw")  # raw P.862
```

`pesq` 套件回傳的是 MOS-LQO，kudio 再透過 P.862.1 的對應關係反推回原始分數；
公式寫在 `pesq` 的 API 參考裡。兩種尺度之間不是差一個固定值。原始分數 1.0
對應窄頻 MOS-LQO 1.16，2.0 是 1.63，3.0 是 2.82，4.0 是 4.15。兩者在原始分數
1.22 和 3.37 處交叉，在這兩點之間原始分數比較高，所以拿窄頻 MOS-LQO 去和論文的
原始 PESQ 比，會讓方法看起來比實際還差。P.862.2 的對應用的是它自己的係數，
所以上面這些都不能拿來換算寬頻分數。

P.862.2 本身不建議把寬頻延伸用在訊號輸入點和擷取點之間有雜訊抑制的系統
（它的第 1 節），而增強後的輸出正是經過這種處理。增強論文裡的寬頻 PESQ
是一種慣例，不是這份建議書背書的用法。

`pesq` 套件的寬頻 PESQ，也就是 kudio 的，是 2005 年原樣發布的參考程式碼。ITU 在
2018 年的 P.862 Corrigendum 2 更正了它：程式送進響度模型的寬頻訊號位準不對，所以
預測出來的分數平均比聽者低 0.8 MOS。更正的做法是把寬頻輸入濾波器乘上 0.251188，
讓訊號落在 76 dB SPL。在 ITU 自己為 Supplement 23 的 1736 對寬頻訊號列的表上，
更正後的分數平均高 0.73，個別的一對從低 0.08 到高 1.40 都有。Torcoli 等人認為
最常被使用的是未更正的版本，並發布了更正過的套件 `pesqc2`[^torcoli]。用它算的寬頻分數，不能和
`pesq` 算的並列比較。

## Hu 與 Loizou 的指標 {#hu-and-loizous-measures}

語音增強的論文常在 PESQ 旁邊再列一組指標：對數似然比（LLR）、Itakura-Saito
距離（IS）、倒頻譜距離（CEP）、加權頻譜斜率（WSS）、頻率加權分段 SNR
（fwSegSNR），以及由 LLR、WSS、一個分段 SNR 和 PESQ 組成的綜合指標 CSIG、CBAK
和 COVL。[^hl08] kudio 照 Loizou 隨這份研究公開的 MATLAB 程式碼來算，發表的綜合指標
數字就是那份程式碼算出來的。程式碼和論文不一致的地方列在模組的 docstring 裡，kudio
和這份程式碼的 Python 移植版吻合到什麼程度，寫在[指標的驗證方式](../validation.md)。
它們只需要 numpy，而且只定義在 8 和 16 kHz。每一個都是一個 30 ms 音框對一個 30 ms
音框地比，所以要像算 SNR 時一樣，先把兩者對齊。

`score` 要指名才會跑它們，名稱是 `llr`、`is`（Itakura-Saito）、`cep`、`wss`
和 `fwsegsnr`，每個也都是獨立的函式：
[`llr`][kudio.core.quality.llr]、
[`itakura_saito`][kudio.core.quality.itakura_saito]、
[`cepstral_distance`][kudio.core.quality.cepstral_distance]、
[`wss`][kudio.core.quality.wss] 和
[`fw_segmental_snr`][kudio.core.quality.fw_segmental_snr]。

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

前四個越低越好，fwSegSNR 越高越好，而它們對 log-MMSE 的看法並不一致：LLR 和
fwSegSNR 說它有幫助，IS 和 WSS 說它變差了，CEP 幾乎沒動。LLR、IS 和 CEP 看的是
每個音框的 LPC 包絡，WSS 看 25 個頻帶之間的斜率，fwSegSNR 看頻譜在這些頻帶之間
怎麼分配，所以各自看到降噪器所做的事的不同部分。讀這些數字時有三件事要記得：

- LLR、IS、CEP 和 WSS 只平均最好的 95% 音框，所以集中在檔案不到 5% 範圍內的損傷，
  可能完全不影響它們。
- 25 個頻帶在兩種取樣率下都到 3.8 kHz 為止，所以就算是 16 kHz 的檔案，WSS 也完全
  看不到那以上的東西。fwSegSNR 的頻帶也停在那裡，但它會先把兩邊的頻譜，在到取樣率
  一半為止的所有頻率點上縮放成面積為 1，所以 4 kHz 以上的雜訊或缺損還是會讓它下降。
- 很小聲的一對，WSS 會隨音量改變。Loizou 的程式把低於滿刻度 -100 dB 的頻帶能量
  一律當成 -100 dB，所以乾淨訊號的峰值低於約 -50 dBFS 之後，較安靜的頻帶就落在這個
  下限上：測試用的那一對（8 kHz）在峰值 -6 到 -46 dBFS 之間都是 35.98，到 -66 dBFS 變成
  27.95。發生這種情況時 `wss` 和 `composite` 會發出警告；請先用同一個增益把兩個
  訊號調到正常音量。

### CSIG、CBAK 和 COVL {#csig-cbak-and-covl}

[`composite`][kudio.core.quality.composite] 預測 ITU-T P.835[^p835] 的三個評分，範圍是
1 到 5：CSIG 是訊號失真，CBAK 是背景有多干擾，COVL 是整體品質。它們是把 PESQ、
LLR、WSS 和一個分段 SNR 線性擬合到聽者評分得到的，所以需要 `pesq` 套件，或用
`pesq=` 傳入你自己算好的 PESQ 分數。經由 `score` 時名稱是 `csig`、`cbak` 和
`covl`，三個來自同一次呼叫。

放進去的是哪一種 PESQ，三個分數都會跟著動。擬合用的是 8 kHz 語音的窄頻 P.862
原始分數，但幾乎所有發表的 16 kHz 數字（包括 VoiceBank-DEMAND）放進去的都是寬頻
MOS-LQO，Loizou 自己 2012 年的程式碼和原本的 pysepm 都是這樣算。`composite` 預設也這樣做：
16 kHz 用寬頻 MOS-LQO，8 kHz 用窄頻原始分數。`mode="nb", scale="raw"` 則是當初擬合
所用的版本：

```python
for kwargs in ({}, {"mode": "nb", "scale": "raw"}):
    c = kudio.composite(clean, enhanced, sr, **kwargs)
    print(f"PESQ {c.pesq:.3f}  CSIG {c.csig:.3f}  CBAK {c.cbak:.3f}  COVL {c.covl:.3f}")
```

```text
PESQ 1.880  CSIG 3.617  CBAK 2.595  COVL 2.715
PESQ 2.875  CSIG 4.217  CBAK 3.071  COVL 3.516
```

同一個輸出，COVL 差了 0.8，所以表格裡要在評分旁邊註明是哪一種 PESQ。`score`
不管 `pesq_scale` 怎麼設，一律用預設。
[`CompositeScore`][kudio.core.quality.CompositeScore] 也帶著四個輸入，所以哪個評分
動了，可以追到是哪個輸入讓它動的。它的 `segsnr` 是綜合指標自己的，用的是加窗的
30 ms 音框，不是 `score` 的 `segsnr`。

## 對數頻譜距離 {#log-spectral-distance}

[`lsd`][kudio.core.evaluator.lsd]（在 `score` 裡叫 `lsd`）逐音框比較兩者的功率
頻譜：對數功率比在頻率上的均方根，再對音框取平均。0 表示頻譜相同，越低越好。
它看得到 SNR 家族平均掉的東西，例如少了一個頻帶：在 2 kHz 低通過的片段得到
18.0，5 dB 的混音是 2.76，log-MMSE 的輸出是 2.42。

預設值給的是 URGENT 2025 挑戰賽評分腳本算出的數字：[^urgent]自然對數、每 16 ms
一個 32 ms 的音框，而且輸出會先乘上它對參考訊號的最小平方增益，所以音量改變不扣分。
頻寬擴展的論文用以 10 為底的對數和它們自己的 STFT；在 44.1 kHz 下，
`scale="log10", n_fft=2048, hop_length=441, match_gain=False, eps=1e-12` 給的就是
NVSR 的 `ssr_eval` 算出的數字。[^nvsr]三種尺度是同一個距離，只差一個常數倍：

```python
low = kudio.lowpass(clean, sr, cutoff=2000)
for scale in ("ln", "log10", "db"):
    print(f"{scale:5s} {kudio.lsd(clean, low, sr, scale=scale):6.2f}")
```

```text
ln     18.00
log10   7.82
db     78.17
```

把你的數字和論文的並列之前，先弄清楚它是哪一種尺度；表格標題寫「dB」，不代表
真的是 dB。增益匹配是投影到參考訊號上，所以兩者必須對齊：晚 1 ms 的片段，有增益
匹配時得到 2.42，`match_gain=False` 時是 0.38。參考訊號是數位靜音的音框，不管
輸出在那裡有什麼，在預設尺度下每個都得 18.4，警告會告訴你這樣的音框有幾個。

## 梅爾倒頻譜失真 {#mel-cepstral-distortion}

[`mcd`][kudio.core.melcepstrum.mcd] 先用動態時間校正把兩者的音框配對，再逐框計算
梅爾倒頻譜之間的距離，單位是 dB，越低越好。各家實作在分析方法、階數、是否計入能量項
c0 以及時間校正方式上都不一樣，所以不同程式算出的 MCD 不能互相比較。這裡的版本是
URGENT 2025 挑戰賽的，[^urgent] 每個設定都照它的評分腳本：SPTK 的梅爾倒頻譜分析、
依取樣率查表得到的階數、計入 c0、半徑 1 的 FastDTW，並先把輸出乘上它對參考訊號的
最小平方增益。因為有時間校正，時間軸不一致的配對也能評，例如語音合成或轉換的輸出；
這時請傳入 `match_gain=False`，因為增益匹配需要兩者逐樣本對齊。

```python
for label, deg in (("noisy", noisy), ("enhanced", enhanced)):
    print(f"{label:8s} MCD {kudio.mcd(clean, deg, sr):.2f} dB")
slower = kudio.time_stretch(clean, 0.9)          # 10% longer
print(f"stretched MCD {kudio.mcd(clean, slower, sr, match_gain=False):.2f} dB")
```

```text
noisy    MCD 6.85 dB
enhanced MCD 5.40 dB
stretched MCD 3.23 dB
```

有設定的取樣率是 8、16、22.05、24、32、44.1 和 48 kHz；其他取樣率請自己傳入
`order` 和 `alpha`。它比這一頁其他指標慢：16 kHz 的十分鐘大約要 16 秒。

## 可懂度：NCM 和 CSII {#intelligibility-ncm-and-csii}

[`ncm`][kudio.core.intelligibility.ncm] 和
[`csii`][kudio.core.intelligibility.csii] 和 STOI 一樣，估計聽的人還能聽懂多少語音，
範圍 0 到 1，越高越好。Ma、Hu 和 Loizou 拿這兩個指標和聽者對降噪後語音的評分比對過，
共 72 種條件，雜訊有汽車、多人交談（babble）、火車和街道。[^mhl09] NCM 是正規化
共變異數指標（normalized covariance metric），在 20 個頻帶裡計算兩個訊號包絡的相關。
CSII 是 Kates 和 Arehart 的同調語音可懂度指數（coherence speech intelligibility
index），[^ka05] 在 SII 的臨界頻帶裡量輸出有多少和參考訊號保持同調（CSII.m 只用
中心頻率到 3.4 kHz 的 16 個），並把參考訊號的大聲、中等和安靜音框分開計算。
Kates 和 Arehart 再把三個值合成一個預測；CSII.m 停在三個值，kudio 也一樣，
所以回傳三個數字。兩者都照 Loizou 的 MATLAB 程式
NCM.m 和 CSII.m 實作，pysepm 移植的也是這兩個檔案，只需要 numpy 和 scipy。

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

`high` 涵蓋大於或等於參考訊號整體 RMS 的音框，`mid` 涵蓋比它低 10 dB 以內的音框，
`low` 則是再往下的所有音框，停頓也算在內。Kates 和 Arehart 的低位準只到低 30 dB
為止；CSII.m 沒有下限，我照程式碼。和 ESTOI 一樣，兩者給 log-MMSE 輸出的
分數，都比它原本的含雜訊輸入低。透過 `score` 時名稱是 `ncm`、`csii_high`、`csii_mid`
和 `csii_low`；CSII 的三級來自同一次呼叫。

要注意的地方：

- NCM 只定義在 8 和 16 kHz，頻帶也隨取樣率而變：8 kHz 時是 300 到 3400 Hz，16 kHz
  時是 300 到 7400 Hz。同一對在兩種取樣率下的分數不同，所以數字旁邊要附上取樣率。
  CSII 接受 8 kHz 以上的任何取樣率，它的頻帶在每種取樣率下都停在 3.7 kHz，所以
  4 kHz 以上發生的事，它幾乎看不到：

    ```python
    from scipy.signal import resample_poly

    clean8, noisy8 = resample_poly(clean, 1, 2), resample_poly(noisy, 1, 2)
    print(f"NCM {kudio.ncm(clean, noisy, sr):.3f} at 16 kHz, "
          f"{kudio.ncm(clean8, noisy8, 8000):.3f} at 8 kHz")
    ```

    ```text
    NCM 0.574 at 16 kHz, 0.663 at 8 kHz
    ```

- 兩者都聽不出音量，也沒有聽閾。在 2 kHz 低通過的片段，每一級的 CSII 仍然是 1.000，
  NCM 是 0.810：它們說的是輸出保留了多少語音的結構，不是聽的人聽不聽得到。
- 兩者都需要先對齊，CSII 最明顯。晚 5 ms 的乾淨片段，高位準 CSII 得 0.757、NCM 得
  0.948；晚 10 ms 則是 0.575 和 0.833。
- 短片段的分數偏高，因為共變異數和同調都是在片段上估計的。一段不相干的合成語音，
  對 1 s 的片段平均得 NCM 0.36，對 3 s 是 0.10，對 10 s 是 0.02。請以整句為單位評分，
  並且只比較長度相近的片段。
- 零不是中性的。兩個訊號都補零時，補的零會算成一致，讓 NCM 變高；在 CSII 裡，它們
  落在低位準，讓分數變低。
- CSII 一路看到 0 Hz，所以參考訊號帶有直流偏移時，輸出裡的數位靜音也會拿到分數。
  先拿 `np.zeros_like(clean)` 評一次，就知道手上的參考訊號有沒有這個問題；如果有，
  把參考訊號以 20 Hz 高通濾波，正向、反向各跑一次，評分前再把輸出減去它自己的
  平均值：

    ```python
    from scipy.signal import butter, sosfiltfilt

    muted = np.zeros_like(clean)
    offset = clean + 0.001 * np.abs(clean).max()        # DC at 0.1% of the peak
    sos = butter(2, 20, "highpass", fs=sr, output="sos")
    for ref in (clean, offset, sosfiltfilt(sos, offset)):
        print(f"{kudio.csii(ref, muted, sr).low:.3f}")
    ```

    ```text
    0.000
    0.255
    0.000
    ```

    只濾參考訊號就好：濾波器會在輸出的數位靜音裡留下振鈴，改變它的分數。

## 以位元計的可懂度：SIIB {#intelligibility-in-bits-siib}

[`siib`][kudio.core.siib.siib] 和 [`siib_gauss`][kudio.core.siib.siib_gauss]
從資訊理論來看可懂度。Van Kuyk、Kleijn 和 Hendriks 把說和聽看成一條鏈：從訊息，
到說話者的語音，再到傳進聽者耳裡的聲音；他們評的是乾淨語音和輸出共有多少資訊，
單位是每秒位元數。[^vkh18a] 兩個訊號都先轉成聽覺對數頻譜，也就是 100 到 6500 Hz
的 28 個 gammatone 頻帶，再加上前向遮蔽；每次取 15 個音框（187.5 ms），轉成 420 個
接近獨立的通道，再把各通道的資訊量加總。SIIB 用 Kraskov、Stögbauer 和 Grassberger
的最近鄰估計器估計每個通道的資訊量。[^ksg04] SIIB^Gauss 出自作者後來對侵入式指標
的評估，[^vkh18b] 把每個通道當成高斯通道，只需要它的相關係數，所以完全不用亂數，
在下面這一對上快了大約 25 倍。論文和作者的 MATLAB 程式 SIIB.m、SIIB_Gauss.m
不一致的地方，kudio 照程式碼；[API 參考][kudio.core.siib]裡這個模組的說明列出每一處
差異，以及它讓分數移動多少。兩者都只需要 numpy 和 scipy。

這一頁的 4 s 片段太短，兩者都評不了：SIIB 需要 436 個語音音框，也就是去掉參考訊號
的靜音後還有 5.45 s，所以這個片段得到 nan，並有一則警告說明原因。下面改用 30 s、
和這個片段同類的語句，加上白雜訊：

```python
gen = np.random.default_rng(1)
talk = 0.01 * gen.standard_normal(30 * sr)        # 30 s of room tone
at = sr // 4
while at < 28 * sr:                               # phrases like the clip's
    phrase = burst(gen.uniform(0.6, 1.4), gen.uniform(100, 220))
    talk[at:at + len(phrase)] += phrase
    at += len(phrase) + int(gen.uniform(0.2, 0.5) * sr)
hiss = gen.standard_normal(len(talk))

for snr in (-5, 0, 5, 10):
    heard = kudio.add_noise_snr(talk, hiss, snr_db=snr, seed=0)
    print(f"{snr:3d} dB    SIIB {kudio.siib(talk, heard, sr):6.1f}  "
          f"SIIB^Gauss {kudio.siib_gauss(talk, heard, sr):6.1f}")

heard = kudio.add_noise_snr(talk, hiss, snr_db=0, seed=0)
denoised = kudio.spectral_enhance(heard, sr)      # log-MMSE, from 0 dB
print(f"denoised  SIIB {kudio.siib(talk, denoised, sr):6.1f}  "
      f"SIIB^Gauss {kudio.siib_gauss(talk, denoised, sr):6.1f}")
print(f"copy      SIIB {kudio.siib(talk, talk, sr):.2f}")
```

```text
 -5 dB    SIIB   34.1  SIIB^Gauss   17.7
  0 dB    SIIB   46.7  SIIB^Gauss   25.4
  5 dB    SIIB   66.1  SIIB^Gauss   40.5
 10 dB    SIIB  123.0  SIIB^Gauss   75.2
denoised  SIIB   50.1  SIIB^Gauss   21.3
copy      SIIB 1335.76
```

分數的範圍不是 0 到 1。同一句話不會有兩次念得一模一樣，作者把這件事建模成每個通道裡
訊息和語音之間 0.75 的相關；隔著它，一個通道最多只能帶 0.596 位元，參考訊號的複本
在兩個指標上都得到上限 1335.76 bit/s。在作者的資料上，聽者完全聽懂的語音大約是
150 bit/s。分數要經過一條對聽測擬合的對應（他們的 Eq. 13）才說得出可懂度，而兩個
指標各要各的對應。兩者也不會一起變動：這裡 SIIB^Gauss 只有 SIIB 的一半多一點，對
log-MMSE 輸出的看法也相反：SIIB 給它的分數高於它所來自的 0 dB 輸入，SIIB^Gauss
則給得比輸入低。透過 `score` 時名稱是 `siib` 和 `siib_gauss`；兩者都要指名才會跑。

要注意的地方：

- 它需要語音，而且要夠多。把比參考訊號響亮端低 40 dB 以上的音框去掉後，語音不到
  5.45 s（436 個音框）的一對，得到 nan 和一則警告。420 維的共變異數要有 421 個堆疊
  才會滿秩；少於這個數量時 SIIB.m 還是會給數字，但其中一部分來自特徵值分解任意挑選
  的特徵向量，所以會隨它所用的 LAPACK 而變。這只是下限，不代表夠用：SIIB.m 的說明
  建議 20 s 以上的語音，把短的語句串接起來，評估論文也是對每個條件這樣做。[^vkh18b]
  長度也會進到分數裡。kudio 的測試用 36 s 的合成音節，中間的停頓會被 40 dB 門檻
  去掉；加上 0 dB 白雜訊時，前 12 s 的 SIIB 是 102.2，整整 36 s 是 124.1。每個條件
  請用長度相近的片段評分。
- 先對齊。兩者的音框是一對一比較的：上面的 talk 晚 5 ms 時，SIIB 得 700.1、
  SIIB^Gauss 得 371.0；晚 10 ms 是 246.1 和 134.5；晚 20 ms 是 100.5 和 53.7，
  比 10 dB 那一對還低。請先用 [`kudio.align`][kudio.core.align.align]。
- 它聽得出音量。遮蔽的下限來自參考訊號，也套用在輸出的頻譜上，所以音量不同的輸出
  就不再算複本：兩倍音量時，talk 的 SIIB 仍然是上限 1335.76，SIIB^Gauss 卻只得
  1112.1；一半音量時，SIIB 得 1184.8、SIIB^Gauss 得 700.9。在雜訊裡影響比較小：
  0 dB 那一對放大兩倍後得 46.2 和 25.3，原本是 46.7 和 25.4。NCM 和 CSII 完全聽不出
  音量。
- 只有參考訊號在說話的音框才算數。音框是依參考訊號挑的，輸出在其他音框裡有什麼，
  完全不看，所以增強器留在停頓裡的雜訊不會被扣分：在測試音節上，只在複本的停頓裡
  加入比語音大 7.5 dB 的白雜訊，SIIB 仍然是上限。
- 沒有任何一步會去掉直流。直流偏移會穿過窗函數進入最低的七個頻帶，也會抬高每個
  音框的位準，可能把參考訊號的停頓抬過 40 dB 門檻，讓它們算進分數。在 0 dB 的測試
  音節上，兩個訊號都加上 0.005 的偏移（比語音的 RMS 低 18 dB），原本保留 1645 個
  音框，變成 2878 個全部保留，SIIB 從 124.1 掉到 72.8，SIIB^Gauss 從 64.0 掉到
  36.8。錄音帶有偏移時，評分前先把平均值減掉。
- 其他取樣率會先轉到 16 kHz，用的是 MATLAB 的 `resample` 設計的濾波器，和 SIIB.m
  一樣。0 dB 那一對先轉成 8 kHz，SIIB 得 50.3、SIIB^Gauss 得 25.6；轉成 44.1 kHz
  是 48.5 和 25.4；16 kHz 時是 46.7 和 25.4。SIIB 會受這個濾波器影響。測試音節用
  近乎理想的濾波器做成 8 和 20 kHz，加上 -10 到 20 dB 的白雜訊，換成更陡的濾波器時，
  SIIB 最多變動 5.1 bit/s，SIIB^Gauss 1.6；8 kHz 的語音改用 `resample_poly` 自己的
  濾波器做成時，則是 1.9 和 0.9；`tools/validation/siib.py` 會跑這組掃描。用別的方式
  重新取樣的實作，在 16 kHz 以外的取樣率下，和 kudio 可能差到幾 bit/s。Python 移植版
  pySIIB 就是一例：在這組掃描上，它給出的是更陡那個濾波器的分數，差在 1e-3 bit/s 以內；
  在 16 kHz 時，它的 SIIB^Gauss 和 kudio 的相對差不到 1e-12，SIIB 最多差 1.4e-5。分數旁邊請附上
  取樣率。
- SIIB 的近鄰數會隨語音變多，所以時間增加得比語音長度快：上面的 30 s 要 2.6 s，
  兩分鐘要半分鐘。SIIB^Gauss 在這裡要 0.1 s，時間隨語音長度成長。

## 多個音源：SI-SDRi、PIT 和 BSS Eval {#several-sources-si-sdri-pit-and-bss-eval}

增強和分離的論文除了分數本身，也會報告系統比輸入進步了多少：SI-SDRi 是輸出的
SI-SDR 減去它所來自的混合訊號的 SI-SDR，SDRi 則是用 BSS Eval 的 SDR 做同樣的
事。[^convtasnet]
[`si_sdr_improvement`][kudio.core.separation.si_sdr_improvement] 和
[`sdr_improvement`][kudio.core.separation.sdr_improvement] 依序接受參考訊號、
輸出和輸入；兩者都不需要額外的套件。

```python
print(f"{kudio.si_sdr_improvement(clean, enhanced, noisy):.2f}")
print(f"{kudio.sdr_improvement(clean, enhanced, noisy):.2f}")
```

```text
6.76
8.32
```

分離系統的輸出沒有固定順序，所以評分之前，要先把每個輸出配對到它對應的音源。
[`pit_si_sdr`][kudio.core.separation.pit_si_sdr] 接受 `(sources, samples)`
形狀的音源和輸出，對每一種配對評分，保留平均 SI-SDR 最高的那一組，也就是
語句層級的排列不變訓練所用的準則；[^upit] 給了混合訊號時，也一併算出進步量。
[`bss_eval`][kudio.core.separation.bss_eval] 給出 BSS Eval 的 SDR、SIR 和
SAR，[^bss] 數字和 mir_eval 的 `bss_eval_sources` 完全相同。

```python
other = 0.01 * rng.standard_normal(4 * sr)       # a second talker, higher
other[16000:48000] += burst(2.0, 210)
sources = np.stack([clean, other])
mixture = sources.sum(axis=0)
hiss = 0.005 * rng.standard_normal((2, 4 * sr))
separated = np.stack([other + 0.3 * clean,       # in the wrong order, each
                      clean + 0.1 * other]) + hiss   # with some of the other

s = kudio.pit_si_sdr(sources, separated, mixture)
print("matched:", s.permutation)
for name, values in (("SI-SDR", s.si_sdr), ("SI-SDRi", s.si_sdri)):
    print(f"{name:8s}", "  ".join(f"{v:6.2f}" for v in values))

b = kudio.bss_eval(sources, separated, mixture, permutation=s.permutation)
for name, label in (("sdr", "SDR"), ("sir", "SIR"), ("sar", "SAR"), ("sdri", "SDRi")):
    print(f"{label:8s}", "  ".join(f"{v:6.2f}" for v in getattr(b, name)))
```

```text
matched: (1, 0)
SI-SDR    18.58    9.60
SI-SDRi   17.83   10.13
SDR       18.63    9.68
SIR       20.67    9.92
SAR       22.93   22.76
SDRi      17.76   10.06
```

`matched` 說明每個音源配到的是哪一個輸出：這裡第二個輸出是第一位說話者。每個
數字都依音源的順序排列，論文報告的是它們的平均 `np.mean(s.si_sdri)`，再對整個
測試集平均。`bss_eval` 自己配對時，照 mir_eval 的做法選平均 SIR 最高的排列；
傳入 `permutation=s.permutation`，整張表就用同一組配對。SI-SDR 會先把兩個訊號的
平均值去掉，和 Conv-TasNet 的 SI-SNR 一樣；torchmetrics 的 SI-SDR 只有在
`zero_mean=True` 時才這麼做，所以有直流偏移的輸出在那裡會得到不同的分數。

## 無法評分的輸入 {#input-a-metric-cannot-score}

這一頁的每個指標都遵守同一條規則。格式不對的輸入會引發 `ValueError`，訊息以函式
名稱開頭，例如 `llr(): ...`，並說明該怎麼做。格式正確、卻沒有東西可評的輸入，
得到 nan 和一則 `UserWarning`，內容是 `<function>(): <reason>; scored nan`，指向
呼叫 kudio 的那一行程式，經由 `score` 呼叫時也一樣。`AudioEvaluate` 會把這樣的
nan 排除在平均之外。

每個指標接受的訊號都一樣：一個聲道的實數，可以是向量、一欄 `(n, 1)` 或一列
`(1, n)`；整數樣本照它型別的滿刻度讀，int16 除以 32768，int32 除以 2^31，和
soundfile 讀檔的方式相同。int64 和 uint64 會被拒絕：沒有音訊檔存這種型別，是
numpy 把整數 list 或 `astype(int)` 轉成的，而照它們的滿刻度讀，16 位元的樣本會低
288 dB，那裡由 1e-12 保護項決定分數。複數、布林和文字陣列，以及被遮罩的樣本也會被
拒絕。取樣率必須是一個正數；`np.array(16000)` 和 `16000.0` 都是 16000，`score`
也照你給的值往下傳。峰值低於 16 位元音訊一階（-90.3 dBFS）的訊號，照常評分並發出
一則警告：int16 樣本放在 int32 陣列裡就會落在那裡，而那麼低的音量下，`snr`、
`si_sdr`、`sdi`、`segmental_snr`、`lsd`、`itakura_saito` 和 `wss` 的固定下限會
改變它們的分數。

| 指標 | 引發 `ValueError` 的情況 | 得到 nan 和警告的情況 |
|---|---|---|
| `snr`、`si_sdr`、`segmental_snr`、`sdi` | 含 NaN 或 inf 的樣本、兩個聲道；`segmental_snr` 另有小於 1 的 `frame_length`，或不在 1 到 `frame_length` 之間的 `hop_length` | 空的一對、整段都是數位靜音的參考訊號（`si_sdr` 是任何常數的參考訊號：靜音、只有直流偏移、只有一個樣本）、大到在 float64 裡平方會溢位的一對；`segmental_snr` 另有比一個音框還短的一對 |
| `pesq` | NaN 或 inf、兩個聲道、8 和 16 kHz 以外的取樣率、PESQ 給不出來的模式和尺度組合（8 kHz 的寬頻、原始的寬頻分數） | 其中一邊是空的、靜音的參考訊號、其中一邊短於 0.25 s、在參考訊號裡找不到語句、輸出是或接近數位靜音 |
| `lsd` | 含 NaN 或 inf 的樣本、兩個聲道、不是正數 Hz 的取樣率 | 空的一對、比一個音框還短的一對（預設 32 ms）、整段都是數位靜音的參考訊號、平方會溢位的一對 |
| `llr`、`itakura_saito`、`cepstral_distance`、`wss`、`fw_segmental_snr` | NaN 或 inf、兩個聲道、8 和 16 kHz 以外的取樣率 | 短於 37.5 ms 的一對（8 kHz 為 300 個樣本，16 kHz 為 600）、靜音的參考訊號 |
| `composite` | 同上、超出尺度範圍的 `pesq=`、`mode="wb", scale="raw"` | 同上，每個欄位都是；`pesq` 套件評不了這一對時（短於 0.25 s、找不到語句、輸出是靜音），只有 CSIG、CBAK、COVL 和 PESQ |
| `ncm` | NaN 或 inf、兩個聲道、8 和 16 kHz 以外的取樣率 | 空的一對、短於 62.6 ms 的一對（8 kHz 為 501 個樣本，16 kHz 為 1001）、靜音的參考訊號 |
| `csii` | NaN 或 inf、兩個聲道、低於 8 kHz 的取樣率 | 每一級：空的一對、短於 45 ms 的一對（8 kHz 為 360 個樣本，16 kHz 為 720）、靜音的參考訊號；單一級：那一級的音框少於兩個 |
| `siib`、`siib_gauss` | NaN 或 inf、兩個聲道、不是整數 Hz 或低於 4000 Hz 的取樣率（以 kHz 給的取樣率） | 空的一對、短於 5.46 s 的一對（16 kHz 為 87,401 個樣本）、去掉參考訊號的靜音後不到 436 個語音音框、從頭到尾都是數位靜音或同一個值的參考訊號、大聲到平方會溢位的輸出 |
| `score` 裡的 `stoi`、`estoi`、`sdr` | 兩個聲道 | 空的一對；任一個訊號整段都是數位靜音；平方會溢位的一對；STOI 和 ESTOI 另有短於 25.6 ms，或 pystoi 去掉靜音音框後剩不到約 0.4 s 的語音；SDR 另有 `bss_eval_sources` 算成 inf 的一對 |
| `score` | NaN 或 inf、兩個聲道、長度差超過上限（只要求 PESQ 時除外）、不認得的指標名稱 | 空的一對，所有要求的指標合起來只有一則警告 |
| [`srmr`][kudio.core.srmr.srmr] | NaN 或 inf、多於一個聲道、8 和 16 kHz 以外的取樣率 | 空的或靜音的片段、切掉靜音後比一個 256 ms 視窗還短的片段 |
| [`wada_snr`][kudio.core.wada.wada_snr] | NaN 或 inf、多於一個聲道、不是正數的取樣率 | 空的片段、短於 1 s 的片段、連續 10 ms 以上的數位靜音合計超過 1%、SNR 落在曲線涵蓋的 -20 到 100 dB 之外 |
| [`dnsmos`][kudio.core.dnsmos.dnsmos]、[`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808] | NaN 或 inf、多於一個聲道、不是正數的取樣率 | 空的片段 |
| [`wer`、`cer`](task-metrics.md#word-and-character-error-rates) | 有一邊不是文字、兩邊配不成對 | 參考那一邊沒有任何詞，`cer` 則是沒有任何字元 |
| [`eer`、`min_dcf`](task-metrics.md#speaker-verification-eer-minimum-dcf-and-the-det-curve) | 分數含 NaN 或 inf、標籤不是布林值或 0/1、被遮罩的試驗、`p_target` 不在 (0, 1) 之間 | 空的清單、沒有目標試驗或沒有非目標試驗的清單 |
| [F0 指標](task-metrics.md#pitch-trackers-gpe-vde-ffe-fpe-and-the-rmse-in-cents) | 長度不同的軌跡、負值或無限大、被遮罩的音框 | 空的軌跡；沒有任何音框兩邊都有聲時的 GPE、FPE 和 RMSE |

只有輸出是數位靜音時，不論是整段還是一部分，這些指標都照常評分而不拒絕，因為輸出
斷掉本身就是瑕疵，分數應該反映出來。整段都是常數的輸出，`si_sdr` 給 0 dB，也就是
它的 1e-12 保護項把 0 / 0 算成的值，並發出警告說明。PESQ（包括 `composite` 裡的）、
STOI、ESTOI 和 SDR 是表格列出的例外，因為它們的套件評不了靜音的輸出；在
`AudioEvaluate` 裡，這樣的檔案會在兩邊的平均中都被排除，而不是把平均拉低。每個需要
參考訊號的指標，都會照[長度不同時](#when-the-lengths-differ)說的方式警告和拒絕。這條
規則怎麼測試，寫在
[指標的驗證方式](../validation.md#the-input-rule-and-the-import-without-optional-packages)。

## 整個測試集 {#a-whole-test-set}

論文表格要用的數字，可以交給 [`score_dataset`][kudio.core.dataset_score.score_dataset]，
或在 shell 裡用 [`kudio score`](../cli.md#score)，一次做完：依 id 配對檔案，跑 `score`
的任何指標和不需要參考訊號的指標，並給出每個平均和它的信賴區間、語句數、
[簽章](#signing-a-result)，以及要引用的文獻。[命令列](../cli.md#score)的那個例子，
在 Python 裡是這樣，其中的 `testset/` 由那一頁的[範例檔案](../cli.md#the-example-files)產生：

```python
import re

result = kudio.score_dataset("testset/enhanced", reference="testset/clean",
                             baseline="testset/noisy", metrics=["pesq", "stoi", "si_sdr"],
                             groups=re.compile(r"^(p\d+)_"))
print(result.to_markdown())          # or to_latex(), to_json()
result.save_rows("scores.csv")       # one row per utterance
```

`groups` 接受編譯好的 pattern（第一個群組就是說話者）、dict，或從 id 對到說話者的
函式；一般的字串會被當成說話者清單的檔名，可以是 Kaldi 的 `utt2spk`，或有 `id` 和
`speaker` 兩欄的 CSV。印出來的表格和每一欄的意思，都在那一頁。

[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] 替檔案清單評分：乾淨的參考訊號、
含雜訊的輸入，以及每個方法一份清單，依位置配對。`clean_files[i]`、`noisy_files[i]`
和每個方法的第 `i` 個檔案，必須是同一段語句。它回報 PESQ、STOI 和 SDR 中有安裝的那些，
表格則需要 pandas（`kudio[data]`）。

```python
files = {"clean": [], "noisy": [], "logmmse": [], "wiener": []}
snr_of = {}
for snr_db in (0, 5, 10):
    mix = kudio.add_noise_snr(clean, noise, snr_db=snr_db, seed=0)
    takes = {"clean": clean, "noisy": mix,
             "logmmse": kudio.spectral_enhance(mix, sr, "logmmse"),
             "wiener": kudio.spectral_enhance(mix, sr, "wiener")}
    for name, wave in takes.items():
        path = f"eval/{name}_{snr_db}dB.wav"
        kudio.save_wave(path, wave, sr)          # creates eval/
        files[name].append(path)
    snr_of[files["noisy"][-1]] = snr_db

ev = kudio.AudioEvaluate(files["clean"], files["noisy"],
                         pesq_mode="nb", pesq_scale="raw")
ev.eval("logmmse", files["logmmse"])
ev.eval("wiener", files["wiener"])
print(ev.get_df())
```

```text
  Enh_method    STOI      SDR
0    logmmse  0.6002  13.1913
1     wiener  0.5990  13.3809
2   Baseline  0.6060   5.0117
```

第一次 `eval` 也會拿同樣的參考訊號，替含雜訊的檔案評一次分，
[`get_df`][kudio.core.evaluator.AudioEvaluate.get_df] 把它們放在最後，作為 Baseline
那一列。每個值都是所有檔案的平均，取到小數第四位。裝了 `pesq` 套件時，
表格會多一欄 PESQ；這次執行沒有裝。每一輪也會在 stderr 畫一條 tqdm 進度條，
怎麼關掉請看[入門](../getting-started.md)。

某個指標對某個檔案沒有分數時，那個檔案在那一欄得到 NaN，並在兩邊的平均中都被
排除，而不是讓整欄都失效；每一輪結束後，每一欄會有一則警告列出這些檔案和原因。
完全無法評分的檔案，例如讀不了、取樣率和參考訊號不同、超過長度上限，或含有 NaN 或
inf，每一欄都是 NaN，並有一則寫出檔名的警告，清單裡其餘的檔案照樣評分。Baseline 那一列是對最後一個方法評過的檔案取平均，所以有 NaN 檔案的方法，
請在 `per_file` 裡和 Baseline 比較，不要在這張表裡比。

[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] 裡，每個方法的每個檔案
各一列，包含 Baseline，分數旁邊還有 `clean`、`noisy` 和被評分的 `file` 路徑，
所以可以依你對每個檔案所知道的任何資訊，把結果拆開來看：

```python
table = ev.per_file()
table["snr_db"] = table["noisy"].map(snr_of)
print(table.pivot(index="snr_db", columns="method", values="SDR").round(2))
```

```text
method  Baseline  logmmse  wiener
snr_db
0           0.01     9.21    9.67
5           5.01    13.33   13.51
10         10.02    17.04   16.96
```

Wiener 在 0 和 5 dB 領先，log-MMSE 則在 10 dB 領先。用
[`Synthesizer`][kudio.core.synth.Synthesizer] 建立的資料集，可以在 `noisy` 欄上
和它的 manifest 合併，manifest 記錄了每份混音的雜訊、SNR、房間和鏈結
（[建立含雜訊資料集](synthesis.md)）。合併時比較的是路徑字串，所以檔案清單請從
manifest 本身建立，在 [`load_manifest`][kudio.core.dataset.load_manifest] 之後用
`[p.noisy for p in pairs]`，而不是重新列一次資料夾。沒有 pandas 時，`ev.rows`
以普通的 dict 存放各方法的列。這裡沒有 SI-SDR 欄；要算的話，用 `score`
對每一對跑迴圈。

要在一段錄音上並排試 kudio 自己的降噪方法，並附上 SI-SDR 和同樣的 PESQ 選項，
請看[降噪](enhancement.md)裡的
[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers]。
這一頁函式的所有參數，都在 API 參考的[指標與對齊](../reference/metrics.md)。

## 信賴區間與檢定 {#confidence-intervals-and-tests}

測試集上的平均，只是抽到的這一批語句給出的一個數字；兩個系統的差距，要和光是換一批
語句就可能造成的變動放在一起看。[`paired_ci`][kudio.core.stats.paired_ci] 對同一批
語句上兩個系統的平均差距給出 bootstrap 信賴區間，
[`permutation_test`][kudio.core.stats.permutation_test] 給出它的 p 值。把每段語句的
說話者傳給 `groups`，重抽樣和翻轉符號時就以整位說話者為單位。下面的 `pesq_base` 和
`pesq_a` 代表基準系統和待比較系統每段語句的 PESQ，也就是
[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] 裡兩個方法的 `PESQ` 欄；
這裡用亂數產生，範例才能自己跑。

```python
import numpy as np
import kudio

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

寫進論文就是：系統 A 的 PESQ 比基準高 0.106（95% 信賴區間 0.057 到 0.168，以 20 位
說話者做 bootstrap；以說話者為單位的符號翻轉檢定，p = 0.0007）。以語句為單位重抽樣，
區間寬度不到四分之一，因為它把 600 段語句當成 600 次獨立的抽樣，其實它們是 20 個聲音，
系統適合某個聲音，那個聲音的 30 段語句就都適合。在 kudio 的模擬裡，組內相關係數 0.3
的說話者資料上，以語句重抽樣的 95% 區間大約只有一半的時候包含真正的平均，以說話者
重抽樣則是 10 位說話者時 89%、30 位時 92%，見
[指標的驗證方式](../validation.md#confidence-intervals-and-the-sign-flip-test)。
不到 10 組時，`bootstrap_ci` 和 `paired_ci` 會警告區間沒什麼意義；`permutation_test`
則是在組數少到任何 p 值都不可能低於 0.05 時才警告，雙尾檢定是 5 組以下。
VoiceBank-DEMAND 的測試集只有兩位說話者，在他們之上的區間只能落在兩人各自的平均
之間；每個區間旁邊都請寫出說話者人數。

表格裡的數字本身，也就是單一系統的平均，用
[`bootstrap_ci`][kudio.core.stats.bootstrap_ci] 算它的區間，重抽樣的方式相同：

```python
for name, scores in (("baseline", pesq_base), ("system A", pesq_a)):
    m = kudio.bootstrap_ci(scores, groups=speakers)
    print(f"{name:8s} {m.estimate:.3f} [{m.low:.3f}, {m.high:.3f}]")
```

```text
baseline 2.578 [2.412, 2.712]
system A 2.684 [2.519, 2.853]
```

兩個區間大半重疊，進步卻還是很明確。各自的區間都帶著 20 個聲音彼此之間的差異，
成對的區間則把這部分扣掉了，因為兩個系統評的是同一批聲音。兩個系統有沒有差別，
要看差距的區間，不是看各自的區間有沒有重疊。

預設是 10,000 次重抽樣的 BCa 區間，種子為 0，所以重跑會得到同樣的數字。13 組以內
（沒有分組時是 13 段語句以內），檢定會跑完全部的符號組合，13 組是 8,192 種，p 值是
精確的；超過時隨機抽 10,000 種，p 值不會低於 1/10,001。

多個系統各自和同一個基準比較，就是多個檢定；[`holm`][kudio.core.stats.holm] 調整
它們的 p 值，讓這些檢定裡出現任何一個誤判的機率維持在顯著水準以內：

```python
pesq_b = pesq_base + np.repeat(rng.normal(0.03, 0.10, 20), 30) + rng.normal(0.0, 0.1, 600)
p = {name: kudio.permutation_test(scores, pesq_base, groups=speakers).pvalue
     for name, scores in (("A", pesq_a), ("B", pesq_b))}
print({name: round(value, 4) for name, value in p.items()})
print({name: round(value, 4) for name, value in kudio.holm(p).items()})
```

```text
{'A': 0.0007, 'B': 0.2932}
{'A': 0.0014, 'B': 0.2932}
```

錯誤率是錯誤總數除以參考詞總數，不是每段語句錯誤率的平均，所以
[`rate_ci`][kudio.core.stats.rate_ci] 和
[`paired_rate_ci`][kudio.core.stats.paired_rate_ci] 每段語句收兩個計數，每次重抽樣
都把兩者各自重新加總再相除。有轉錄文字時，計數來自對每一對語句執行
[`word_errors`][kudio.core.wer.word_errors]，也就是 `c.errors` 和
`c.reference_length`。給了 `totals` 時，`permutation_test` 回報的是兩個合併錯誤率的
差；`alternative="less"` 問的是系統的錯誤是否比較少：

```python
words = 5 + rng.poisson(10, 600)                   # reference words per utterance
hard = np.repeat(rng.beta(2, 14, 20), 30)          # each speaker's own error rate
errors_base = rng.binomial(words, hard)
errors_a = rng.binomial(words, 0.85 * hard)
wer = kudio.rate_ci(errors_base, words, groups=speakers)
gap = kudio.paired_rate_ci(errors_a, errors_base, words, groups=speakers)
test = kudio.permutation_test(errors_a, errors_base, totals=words,
                              groups=speakers, alternative="less")
print(f"baseline WER {wer.estimate:.2%} [{wer.low:.2%}, {wer.high:.2%}]")
print(f"A minus baseline {gap.estimate:+.2%} [{gap.low:+.2%}, {gap.high:+.2%}], "
      f"p = {test.pvalue:.4f}")
```

```text
baseline WER 13.20% [9.75%, 20.64%]
A minus baseline -2.11% [-3.68%, -1.22%], p = 0.0004
```

每個結果都帶著它的計算依據：`n`、`n_groups`、`n_resamples` 和 `method`，檢定另外有
是否跑完全部符號組合（`exact`）。分數裡有 NaN 會被拒絕：把指標無法評分的語句從每一份
清單裡拿掉，`groups` 也一樣，並寫明拿掉了幾段。每次重抽樣都得到估計值、寬度為零的
區間會附上警告：500 段語句上 WER 為 0，並不能說明下一批 500 段的 WER 會落在哪裡。
為什麼用 BCa、分組檢定的虛無假設怎麼設、p 值怎麼把觀察到的那一種組合算進去，都寫在
API 參考裡 [`kudio.core.stats`][kudio.core.stats] 的模組說明。

## 回報和引用指標 {#reporting-and-citing-a-metric}

kudio 算的每個指標，在 `kudio.METRIC_INFO` 裡都有一筆資料，用 `score` 的名稱或
函式本身的名稱都查得到。[`metric_info`][kudio.core.registry.metric_info] 取出其中一筆：

```python
info = kudio.metric_info("csii_high")
print(info.function, info.field, info.better, info.range)
print(info.validation)
```

```text
kudio.csii high higher (0.0, 1.0)
validation.md#ncm-and-csii
```

`variants` 寫的是會改變數值的設定，也就是表格裡要和數字並列的東西；`refs` 是它
出自哪些文獻。[`cite`][kudio.core.registry.cite] 把一串名稱轉成 BibTeX：最前面是
kudio 自己的條目，接著是各篇文獻，依指標給的順序，每篇只出現一次。`score` 的名稱
和函式名稱都可以用。

```python
print(kudio.cite(["pesq", "stoi"]))
```

這些條目就是這幾頁註腳裡的文獻，重音符號用 LaTeX 的寫法，所以 BibTeX 和
biblatex 都能直接用。

### 為結果加上簽章 {#signing-a-result}

兩張表都寫「PESQ 2.45」，算的不一定是同一個東西。原始 P.862 的 2.0 是窄頻
MOS-LQO 的 1.63；[CSIG、CBAK 和 COVL](#csig-cbak-and-covl) 那一節的輸出，放進
composite 的 PESQ 換一種，COVL 就差 0.80，VoiceBank-DEMAND 的含噪測試集平均差
0.85；min DCF 不寫先驗機率就沒有意義。Post 在 BLEU 上發現同樣的問題，寫了
sacreBLEU，每個分數旁邊都印出一行短短的簽章。[^post]
[`signature`][kudio.core.signature.signature] 替 kudio 做同一件事：給它指標和
它們跑的設定，它寫出一行字，列出每個會改變數值的設定（照實際解析出來的值），
以及算出各個數值的套件版本。拿[先算一個分數](#a-first-score)那段片段來說：

```python
scores = kudio.score(clean, enhanced, sr, ["pesq", "stoi", "si_sdr"])
text = kudio.signature(scores.keys(), sr)
print(text)
```

```text
kudio:4.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5,whole=pesq|si_sdr|pesq:mode=wb,scale=lqo|stoi|pkg:pesq=0.0.4,pystoi=0.4.1
```

這是裝了 pesq 0.0.4 和 pystoi 0.4.1 時的結果。`mode=wb` 是 `pesq_mode='auto'`
在 16 kHz 下實際變成的模式；預設值照它算出來的值寫出來，所以預設值以後改了，
這行字照樣看得懂。`pair` 說明這對訊號事先沒有用 `kudio.align` 對齊（有對齊就傳
`aligned=True`），指標把這對訊號截到較短的長度，並拒絕會讓較長那段超過一半沒被
量到的訊號對（`limit`，即 `MAX_LENGTH_MISMATCH`），而 PESQ 拿到的是完整的兩段。
不管指標用什麼順序傳進來，都照登錄表的順序寫，所以同一組指標和設定一定得到同
一行字。

單獨呼叫的指標，用 dict 傳它的關鍵字參數，也就是呼叫時用的那個 dict：

```python
print(kudio.signature({"eer": {}, "min_dcf": {"p_target": 0.01}}, n=4874))
```

```text
kudio:4.0.0|eer:method=linear|min_dcf:p_target=0.01,c_miss=1.0,c_fa=1.0|n:4874
```

`n` 是評了幾筆，`ci` 是在這些筆數上算的是哪一種信賴區間、怎麼算的：函式名稱放在
`kind`，加上結果回報的方法（BCa 給不出區間時是 `'percentile'`），以及傳給它的
信賴水準、重抽次數、種子和分組。[信賴區間與檢定](#confidence-intervals-and-tests)
那一節的區間，簽起來是這樣：

```python
print(kudio.signature(["pesq"], sr, n=ci.n,
                      ci=dict(kind="paired_ci", method=ci.method, confidence=0.95,
                              n_resamples=10000, seed=0, groups=speakers)))
```

```text
kudio:4.0.0|sr:16000|pair:aligned=no,whole=pesq|pesq:mode=wb,scale=lqo|pkg:pesq=0.0.4|n:600|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=20
```

DNSMOS 那一欄還會寫出模型檔和把輸出對應到 MOS 的多項式，各用 SHA-256 的前 8 個
十六進位數字標明，用兩份不同權重跑出來的結果分得出來。套件版本和模型都是從執行
的這台電腦讀的，所以我在算出數字的那次執行裡就寫簽章。
[`parse_signature`][kudio.core.signature.parse_signature] 把簽章讀回 dict，
方便逐欄比較兩個簽章：

```python
print(kudio.parse_signature(text)["pesq"])
```

```text
{'mode': 'wb', 'scale': 'lqo'}
```

簽章放在表格下方或註腳裡，參考文獻用 `kudio.cite` 產生。

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
