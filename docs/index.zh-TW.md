---
description: >-
  Kudio 是處理語音音訊的 Python 工具組：檔案讀寫、模型特徵、不需參考訊號的檢查、
  房間與電話線路、含雜訊資料集、降噪、品質指標和錄音。
hide:
  - navigation
---

# Kudio

![Kudio, audio processing](assets/banner.webp)

Kudio 是處理音訊的 Python 工具組，主要對象是語音。它能
[讀寫檔案](guide/audio-files.md)，在沒有乾淨版本可比對時
[檢查一段錄音](guide/inspection.md)，找出[哪裡有人在說話、音高多少](guide/speech.md)，
對片段做[剪接、濾波和資料擴增](guide/editing.md)，也能把音訊轉成
[頻譜圖和模型特徵](guide/features.md)。另外它可以照房間、電話線路和有損傳輸的方式
[劣化音訊](guide/degradation.md)，[建立含雜訊的資料集](guide/synthesis.md)並且能用同一個
亂數種子重建，[降噪](guide/enhancement.md)、[評估結果](guide/metrics.md)，以及
[從麥克風錄音](guide/live-audio.md)，邊錄邊畫圖或邊降噪。大部分功能都是吃一個 numpy
陣列加一個取樣率的小函式，寫在一般的腳本裡就能組合起來用，不需要先設定什麼。

我寫它是為了整理語音語料庫，以及訓練和評估降噪模型；它也同樣適合一整個資料夾的
現場錄音，在有人逐一聽完之前，先知道哪些能用。常用的工作也做成了 `kudio` 的
[子指令](cli.md)，從 `kudio report` 到 `kudio synth` 一共十五個。負責檢查的那幾個在
發現問題時會以 1 結束，所以可以直接當 shell 腳本裡的測試。

## 安裝 {#install}

```bash
pip install kudio
pip install "kudio[eval,viz]"    # extras go in brackets, by name
```

核心安裝會帶入 numpy、scipy、librosa、soundfile、PyYAML、tqdm 和 PyWavelets，
需要 Python 3.9 以上。選用套件組有：`[viz]` 畫圖、`[eval]` 算 STOI、ESTOI 和 SDR、
`[pesq]` 算 PESQ、`[dnsmos]` 算 DNSMOS、`[data]` 匯出 dataframe 和 Excel、`[audio]`
錄音和播放。`[all]` 會裝除了 `[pesq]` 以外的全部，`[pesq]` 要編譯 C 程式碼，所以另外裝。
各組帶進哪些套件，列在[入門](getting-started.md)。

## 快速導覽 {#a-short-tour}

一段含雜訊的錄音，經過檢查、降噪再評分。這兩個檔案是合成的一對，也正因為是合成的，
才有乾淨的參考訊號可以拿來評分。請先展開下面的區塊，在空資料夾裡執行一次，把檔案做出來。

??? example "產生 `clean.wav` 和 `noisy.wav`"

    ```python
    import numpy as np
    import kudio

    sr = 16000
    t = np.arange(4 * sr) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)               # a pitch that wanders
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 20))  # 19 harmonics
    syllables = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, None)  # sound, pause, sound...
    clean = (0.1 * voice * syllables).astype(np.float32)

    hiss = np.random.default_rng(7).standard_normal(len(clean))
    noisy = kudio.add_noise_snr(clean, hiss, snr_db=0, seed=0)  # as loud as the voice

    kudio.save_wave("clean.wav", clean, sr)
    kudio.save_wave("noisy.wav", noisy, sr)
    ```

    這是一個音高在 100 到 140 Hz 之間飄移的嗡嗡聲，切成三分之一秒長的音節，中間夾著
    同樣長度的停頓，再用 [`add_noise_snr`][kudio.effects.augment.add_noise_snr]
    混入和人聲同功率的白雜訊。停頓是刻意留的：雜訊估計器是從安靜的地方學雜訊，
    一段完全沒有停頓的片段，能讓它們學的東西很少。

=== "Python"

    ```python
    import kudio

    y, sr = kudio.file_load("noisy.wav")           # float32 mono, at the file's own rate

    report = kudio.audio_report(y, sr)             # level, clipping, noise floor, bandwidth
    print(report)
    for problem in report.problems():
        print(problem)

    denoised = kudio.spectral_enhance(y, sr)       # log-MMSE over an MCRA noise estimate
    kudio.save_wave("denoised.wav", denoised, sr)  # 16-bit PCM unless you ask otherwise

    ref, _ = kudio.file_load("clean.wav")          # the clean half of the pair
    for name, audio in [("noisy", y), ("denoised", denoised)]:
        s = kudio.score(ref, audio, sr, ["si_sdr", "snr"])
        print(f"{name:<9} SI-SDR {s['si_sdr']:+5.1f} dB   SNR {s['snr']:+5.1f} dB")
    ```

    ```text title="輸出"
    4.00s @ 16000 Hz · peak -10.1 dBFS · -22.8 LUFS · SNR~7 dB · bandwidth 8000 Hz
    estimated SNR around 7 dB — the noise floor is close to the content
    noisy     SI-SDR  +0.0 dB   SNR  +0.0 dB
    denoised  SI-SDR  +8.3 dB   SNR  +8.9 dB
    ```

=== "命令列"

    ```bash
    kudio report noisy.wav                        # exits 1: it found something
    ```

    ```text title="輸出"
    4.00s @ 16000 Hz · peak -10.1 dBFS · -22.8 LUFS · SNR~7 dB · bandwidth 8000 Hz

    · estimated SNR around 7 dB — the noise floor is close to the content

    (This is a technical check. A perceptual score with no reference needs a trained model — see kudio.dnsmos.)
    ```

    ```bash
    kudio enhance noisy.wav denoised.wav          # the defaults the Python tab used
    ```

    ```text title="輸出"
    log-MMSE + mcra noise estimate -> denoised.wav
      noise floor -27.3 -> -46.4 dBFS
    ```

    ```bash
    kudio align clean.wav denoised.wav --metrics  # the offset first, then the score
    ```

    ```text title="輸出"
    already aligned, correlation 0.934
    overlap     4.000s of 4.000s / 4.000s

    si_sdr          8.34 dB  as the files sit
                    8.34 dB  aligned
    ```

[`audio_report`][kudio.core.report.audio_report] 只看錄音本身，而真實的錄音你手上也只有
這個；[`problems()`][kudio.core.report.AudioReport.problems] 留下值得處理的發現。它的 SNR
是拿響的音框和安靜的音框比，所以一對以 0 dB 混出來的檔案，它會說 7 dB；差異從哪裡來，
[入門](getting-started.md)有說明。至於錄音聽起來如何，是另一個問題，
[`dnsmos`][kudio.core.dnsmos.dnsmos] 可以在沒有參考訊號的情況下預測。它需要 `[dnsmos]`
選用套件組和 DNS-Challenge 的模型權重，kudio 不附帶這些權重。

註解裡的組合是 [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] 的預設，
也是 28 種組合之一：七種增益規則，每一種都可以搭配四種雜訊估計器中的任何一種。
要替自己的錄音挑哪一種，就是[降噪](guide/enhancement.md)那一頁在談的事。

[`score`][kudio.core.evaluator.score] 量 SI-SDR 和 SNR 的方式，是拿參考訊號的第 *i* 個
樣本和另一個訊號的第 *i* 個樣本比，所以兩者必須對齊。這一對是對齊的：它本來就是這樣
做出來的，而且 `spectral_enhance` 回傳的樣本數和輸入完全一樣，一個都沒移位。
對可能沒對齊的一對，[`align`][kudio.core.align.align] 會找出偏移，把兩者都裁到共同的
部分，命令列分頁裡的 `kudio align --metrics` 就是先做這件事再評分。該用哪個指標、
哪些需要選用套件組，請看[品質評估](guide/metrics.md)。

## 接下來看哪裡 {#where-to-go-next}

<div class="grid cards" markdown>

-   __[入門](getting-started.md)__

    ---

    安裝 kudio 和選用套件組，把一段含雜訊的錄音從檔案一路處理到評分結果，
    以及 kudio 的錯誤和記錄長什麼樣子。

-   __[使用指南](guide/index.md)__

    ---

    一項工作一頁，從讀檔到用麥克風錄音，說明每個工具背後的考量，以及容易踩到的地方。

-   __[命令列](cli.md)__

    ---

    `kudio` 指令和它的子指令，會印出什麼，以及什麼時候以非零值結束。

-   __[API 參考](reference/index.md)__

    ---

    每個公開名稱，依模組排列，附完整簽名和參數，另外有一份 `import kudio`
    給你的所有名稱的索引，依主題分組。

</div>

## kudio 會拒絕什麼，又會默默做什麼 {#what-kudio-refuses-and-what-it-does-quietly}

錯的結果通常看起來跟對的一樣。以下這些地方，kudio 會發出警告或拒絕，而不是交給你一個結果：

- [`score`][kudio.core.evaluator.score]、
  [`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate] 以及不需額外安裝的指標函式，
  在兩個訊號長度不同時會警告；當較長那個有超過一半量不到時，會引發 `ValueError`
  （[`MAX_LENGTH_MISMATCH`][kudio.core.evaluator.MAX_LENGTH_MISMATCH]）。
  [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] 和 `pesq`
  用同樣的方式檢查參考訊號，而且每個指標都會拒絕立體聲的一對，不會把兩個聲道當成
  一個訊號來評分。
- [`pesq`][kudio.core.evaluator.pesq] 遇到 8 和 16 kHz 以外的取樣率會引發
  `ValueError`，不會背著你偷偷重新取樣；遇到標準沒有定義的組合，例如原始的寬頻分數，
  也一樣。
- [`save_wave`][kudio.core.io.save_wave] 拒絕寫入含 NaN 或 inf 的波形，不論格式，
  都以 [`AudioIOError`][kudio.exceptions.AudioIOError] 拒絕。寫成 `FLOAT` 的話，NaN
  會傳到每個讀這個檔案的程式；寫成整數格式則會變成普通的樣本值，下游沒有人會知道這段
  已經壞了。響度函式量這種片段時會得到 `nan` 並發出警告，絕不會量成靜音的 `-inf`，
  `normalize_lufs` 和 `match_loudness` 則會拒絕它。
- [`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] 遇到 `specsub`、
  `multiband`、`spectral_gate` 和 `quantile` 雜訊估計器時，會引發
  [`FeatureError`][kudio.exceptions.FeatureError]。這幾個要一次看整張頻譜圖，
  與其執行一個只是同名的東西，不如直接拒絕。

`ValueError` 不是 [`KudioError`][kudio.exceptions.KudioError]，所以
`except kudio.KudioError` 不會攔到它們。缺少選用套件組時會引發
[`DependencyError`][kudio.exceptions.DependencyError]，它同時也是 `ImportError`，
訊息裡會寫出要執行的 `pip install`。每個例外類別以及由誰引發，都列在[入門](getting-started.md)。

以下這些則是默默發生的：

- [`file_load`][kudio.core.io.file_load] 會把多聲道檔案的各聲道平均成一個，
  除非你傳 `mono=False`。
- 在整數格式中，包括預設的 16-bit PCM，`save_wave` 會把超過 ±1.0 的部分截在滿刻度。
  混音可能太大聲時，請寫成 `subtype="FLOAT"`。
  [`Synthesizer`][kudio.core.synth.Synthesizer] 是例外：混音超過滿刻度時 `syn()`
  會警告，`.clipped` 會列出那些檔案。
- `pesq` 在預設設定下，16 kHz 回報的是寬頻 MOS-LQO。同一個檔案的原始窄頻 P.862 分數
  是另一個數字，就連窄頻 MOS-LQO 也會把原始分數 2.0 換成大約 1.6，所以跟論文比較之前，
  先確認對方引用的是哪一種。用 `mode=` 和 `scale=` 選擇；透過 `score` 和
  `AudioEvaluate` 時則用 `pesq_mode=` 和 `pesq_scale=`。
- kudio 的例行記錄寫到 `kudio.*` 這些 logger，在你的程式設定 `logging` 之前一律不會印出來：
  例如某個檔案即將被覆寫、某個輸出資料夾被清空、清單裡某個項目被略過。音訊本身的問題，
  像是檔案裡有 NaN，或 [`normalize_lufs`][kudio.core.loudness.normalize_lufs] 把峰值推過
  滿刻度，會以 Python 警告出現，不設定也看得到。任何要無人看管執行的東西，我都會在開頭加
  `logging.basicConfig(level=logging.WARNING)`。

凡是會用到亂數的地方都可以傳亂數種子：[混合語料庫](guide/synthesis.md)裡雜訊從哪裡截取、
[房間與通道](guide/degradation.md)裡產生的房間或封包遺失樣式、訓練／驗證／測試的切分。
它們的預設都是 `seed=None`，每次執行都會重新抽一個，所以請自己傳一個。沒有給種子的
`Synthesizer` 會把抽到的種子留在 `.seed`，那份語料庫還是可以重建。

## 相關專案 {#related-projects}

我另外還有兩個專案是建立在 kudio 之上的，兩個都在 PyPI 上。

[KudioStudio](https://github.com/recklight/KudioStudio)（`pip install kudiostudio`）
:   建立在 kudio 上的桌面工作台，視窗由系統本身的網頁元件畫出。打開一個片段，看它、編輯它、
    降噪、量測，並用它建立資料集，都不必先寫腳本。每個控制項都對應一個 kudio 函式，
    每個面板下方的說明會顯示它實際會呼叫的那一行，所以在那裡調好的設定可以直接搬到程式碼裡。

[KudioEnhance](https://github.com/recklight/KudioEnhance)（`pip install kudioenhance`）
:   用 PyTorch 在對數功率頻譜圖上訓練降噪模型：全連接降噪自編碼器、雙向 LSTM、
    擴張摺積自編碼器、因果的摺積遞迴網路，以及二維 U-Net。它的訓練資料由
    [`Synthesizer`][kudio.core.synth.Synthesizer] 混成，訓練好的模型可以放進
    [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers]，和統計方法並列，
    用同一個參考訊號評分。
