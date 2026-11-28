# 降噪 {#denoising}

`kudio.enhance` 是單聲道的統計式降噪：不用模型，不用下載任何東西。它估計雜訊，算出每個
時頻 bin 裡有多少語音，再依此把每個 bin 調小。七種增益規則可以配四種雜訊估計方法中的任何一種，
其中四種規則也能在即時串流上執行。

| 你手上有 | 用 |
|---|---|
| 一段有雜訊的片段 | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| 一段片段，想知道哪個方法適合它 | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| 還在進來的音訊 | [`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] |
| 一整個資料夾的錄音 | [`enhance_folder`][kudio.enhance.spectral.enhance_folder] |

??? example "範例使用的片段"

    16 kHz、五秒：兩個諧波「人聲」，前、中、後都有停頓，整段混上 5 dB SNR 的白雜訊。

    ```python
    import numpy as np

    import kudio

    sr = 16000
    t = np.arange(sr * 5) / sr
    rng = np.random.default_rng(0)

    def voice(start, stop, f0):
        """Thirteen harmonics of f0, rising and falling 3.5 times a second."""
        span = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 14))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
        return np.where(span, tone * swell / 7, 0.0)

    clean = (voice(0.5, 2.0, 140) + voice(2.8, 4.5, 170)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, rng.standard_normal(len(clean)),
                                snr_db=5, seed=0)
    ```

## 一行呼叫 {#one-call}

```python
enhanced = kudio.spectral_enhance(noisy, sr)   # log-MMSE, MCRA noise estimate
print(f"SI-SDR {kudio.si_sdr(clean, noisy):.2f} dB -> {kudio.si_sdr(clean, enhanced):.2f} dB")
```

```text
SI-SDR 4.98 dB -> 12.21 dB
```

![有雜訊的片段和 log-MMSE 輸出在 0 到 4 kHz 的頻譜圖。前兩秒之後，輸出的雜訊低了大約 19 dB，諧波完好；在那之前還留著一條條水平的殘餘雜訊。](../images/enhancement-before-after-light.webp#only-light)
![有雜訊的片段和 log-MMSE 輸出在 0 到 4 kHz 的頻譜圖。前兩秒之後，輸出的雜訊低了大約 19 dB，諧波完好；在那之前還留著一條條水平的殘餘雜訊。](../images/enhancement-before-after-dark.webp#only-dark)

輸出是 float32，長度和輸入相同，逐樣本對齊。音訊必須是單聲道，短於一個音框（`n_fft`，512）
的片段會被拒絕。方法自己的參數用關鍵字傳；方法不收的參數會丟出
[`FeatureError`][kudio.exceptions.FeatureError]，並列出它收的參數。`hop_length=` 預設是
`n_fft` 的四分之一；視窗重疊太少的跳距會被拒絕。

## 增益規則與雜訊估計 {#gain-rules-and-noise-estimators}

每種規則都把雜訊估計 $\lambda_d(k,l)$ 變成介於下限和 1 之間、乘在複數頻譜上的增益，相位
維持有雜訊的那個。`wiener`、`mmse_stsa`、`logmmse` 和 `omlsa` 會經過 Ephraim 和 Malah 的
決策導向先驗 SNR $\xi$[^em84]，$\alpha = 0.98$，$\gamma$ 上限 40 dB，$\xi$ 下限 −25 dB。

| 方法 | 做什麼 | 自己的參數 |
|---|---|---|
| `specsub` | 功率譜相減[^boll]，固定的過度相減[^berouti] 加上下限 | `over_subtraction` 2.0、`spectral_floor` 0.002 |
| `multiband` | 依各頻帶的 SNR 決定相減係數[^kamath] | `bands` 6、`spectral_floor` 0.002 |
| `wiener` | 增益 $\xi/(1+\xi)$ | `alpha` 0.98 |
| `mmse_stsa` | 振幅的 MMSE 估計[^em84] | `alpha` 0.98 |
| `logmmse` | 對數振幅域的 MMSE[^em85]；預設 | `alpha` 0.98 |
| `omlsa` | 以語音存在機率加權的 log-MMSE[^omlsa] | `alpha` 0.98、`q` 0.30 |
| `spectral_gate` | 高於雜訊估計變動幅度的平滑門檻遮罩 | `n_std_thresh` 1.5、`prop_decrease` 1.0、`freq_smooth_hz` 500、`time_smooth_ms` 50 |

除了 `spectral_gate` 都收 `gain_floor_db`，預設 −25 dB：一個 bin 最多能調小多少。
[`METHODS`][kudio.enhance.spectral.METHODS] 存著每個參數的預設、範圍、步距、單位和一句說明，
GUI 可以直接拿來建控制項。

| 估計方法（`noise=`） | 假設 | `noise_kwargs` | 串流 |
|---|---|---|---|
| `mcra`（預設） | 每個 bin 的滑動平均，在 bin 比滑動最小值高 5 倍時停住[^mcra] | `window=100`、`delta=5` | 可 |
| `quantile` | 每個 bin 在整個檔案上的低分位數，放大到穩定雜訊的平均 | `quantile=0.15` | 不可 |
| `initial` | 開頭的音框只有雜訊 | `n_frames=6` | 可 |
| `minimum` | 平滑頻譜的滑動最小值乘上偏差係數 | `window=60`、`bias=1.5`、`smoothing=0.85` | 可 |

```python
for noise in kudio.NOISE_ESTIMATORS:
    out = kudio.spectral_enhance(noisy, sr, "logmmse", noise=noise)
    print(f"{noise:9} {kudio.si_sdr(clean, out):.2f} dB")
```

```text
mcra      12.21 dB
quantile  14.16 dB
initial   14.86 dB
minimum   6.86 dB
```

這裡 `initial` 最好，因為片段開頭剛好是半秒只有雜訊，完全符合它的假設；
[`NOISE_ESTIMATORS`][kudio.enhance.spectral.NOISE_ESTIMATORS] 對每種方法各有一句說明。`mcra`
一開始估得太低，大約要 1.6 s 才穩定，就是圖裡那些條紋。
[`estimate_noise_psd`][kudio.enhance.spectral.estimate_noise_psd]、
[`make_noise_tracker`][kudio.enhance.spectral.make_noise_tracker]、
[`DecisionDirected`][kudio.enhance.spectral.DecisionDirected] 和
[`make_gain_rule`][kudio.enhance.spectral.make_gain_rule] 是組成這些方法的零件，可以拿去寫自己
的迴圈。

## 挑選方法 {#choosing-a-method}

```python
results = kudio.compare_enhancers(noisy, sr, reference=clean,
                                  noises=list(kudio.NOISE_ESTIMATORS), metrics=False)
ranked = sorted((r for r in results if not r.error), key=lambda r: r.si_sdr_db, reverse=True)
print(len(ranked))
for r in ranked[:3]:
    print(f"{r.label:20} SI-SDR {r.si_sdr_db:+.2f} dB  dSNR {r.delta_snr_db:+.2f} dB")
```

```text
28
wiener + initial     SI-SDR +15.37 dB  dSNR +10.48 dB
omlsa + initial      SI-SDR +15.26 dB  dSNR +10.36 dB
logmmse + initial    SI-SDR +14.86 dB  dSNR +10.00 dB
```

[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] 執行要求的每一組（方法、估計方法），
各傳回一個 [`EnhanceResult`][kudio.enhance.spectral.EnhanceResult]：時間、底噪和響度；有
`reference=` 時再加上 SNR、SI-SDR 和 `delta_snr_db`，也就是比輸入多賺的 SNR。預設
`metrics=True` 會加上 STOI，裝了 `kudio[pesq]` 再加上 PESQ。失敗的那一組會帶著 `error` 傳回。
其他關鍵字只交給收它的那幾組；沒有任何一組收的關鍵字，在執行之前就會被拒絕。沒有參考訊號時
只有底噪和響度，而底噪最低的不一定最好。

`(y, sr)` 的函式，單獨給或以 `(name, function)` 給，也能放進同一張表，估計方法欄寫 `own`
（[`CustomEnhancer`][kudio.enhance.spectral.CustomEnhancer]），所以用
[KudioEnhance](https://github.com/recklight/KudioEnhance) 訓練的模型可以和這些方法一起排名。

## 幫不上忙的情況 {#where-it-does-not-help}

在乾淨人聲裡以各種輸入 SNR 混入白雜訊，MCRA 之下的 `delta_snr_db`：

| 輸入 SNR | 最好的規則 | 其他六種裡最差的 | `spectral_gate` |
|---|---|---|---|
| −5 dB | +10.5 | +6.9 | +6.5 |
| 5 dB | +7.6 | +6.0 | +0.4 |
| 15 dB | +3.1 | +0.5 | −9.9 |
| 20 dB | −0.1 | −4.0 | −15.0 |

這些方法是給吵的音訊用的：在 20 dB 時每種規則都讓片段變得更差。它們也需要聽到單獨的雜訊。
沒有停頓時，`logmmse` 配 `mcra` 只多 +1.3 dB，而不是 +7.6；前面加一秒房間底噪、配
`noise="initial"`，可以到 +8.7。

## 還在進來的音訊 {#audio-that-is-still-arriving}

```python
enhancer = kudio.StreamEnhancer(sr, "logmmse")
pieces = [enhancer.process(noisy[i:i + 1024])       # one 64 ms block at a time
          for i in range(0, len(noisy), 1024)]
pieces.append(enhancer.flush())
streamed = np.concatenate(pieces)
print(f"latency {enhancer.latency_seconds * 1000:.0f} ms, {len(noisy)} in, {len(streamed)} out,",
      f"SI-SDR {kudio.si_sdr(clean, streamed):.2f} dB")
```

```text
latency 32 ms, 80000 in, 80000 out, SI-SDR 13.45 dB
```

[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] 用同樣的更新方式，一次處理任意大小的
區塊。延遲是一個音框，`n_fft / sr`，是等待而不是位移，所以輸出的第 *i* 個樣本和輸入的第 *i*
個對齊；[`flush`][kudio.enhance.streaming.StreamEnhancer.flush] 處理最後不滿一個音框的部分，
[`reset`][kudio.enhance.streaming.StreamEnhancer.reset] 為下一段錄音清掉所有狀態。
[`STREAMABLE_METHODS`][kudio.enhance.streaming.STREAMABLE_METHODS] 裡的四種規則能配 `mcra`、
`initial` 或 `minimum` 串流；其他的會丟出 `FeatureError`，並給出離線的呼叫方式。請餵它
[`StreamRecorder.drain()`][kudio.core.stream.StreamRecorder.drain]，不要用 `tail()`，並保持
單聲道（[錄音與即時檢視](live-audio.md)）。

## 整個資料夾 {#a-whole-folder}

```python
hiss = np.random.default_rng(1).standard_normal(len(clean))
for name, snr_db in [("day1/take1.wav", 0), ("day1/take2.wav", 5), ("day2/take1.wav", 10)]:
    kudio.save_wave(f"noisy/{name}", kudio.add_noise_snr(clean, hiss, snr_db=snr_db, seed=0), sr)
result = kudio.enhance_folder("noisy", "enhanced", "logmmse", report=True)
print(result)
for path, before, after in result.floors:
    print(f"{path.as_posix()}: floor {before:.1f} -> {after:.1f} dBFS")
```

```text
3/3 written, 0 failed
enhanced/day1/take1.wav: floor -24.4 -> -43.5 dBFS
enhanced/day1/take2.wav: floor -29.4 -> -48.4 dBFS
enhanced/day2/take1.wav: floor -34.4 -> -53.3 dBFS
```

[`enhance_folder`][kudio.enhance.spectral.enhance_folder] 像 [`convert_folder`][kudio.core.io.convert_folder]
一樣保留子資料夾結構，傳回 [`EnhanceFolderResult`][kudio.enhance.spectral.EnhanceFolderResult]：
有檔案失敗時為假，`report=True` 時附上量到的底噪。方法和參數在第一個檔案之前就檢查一次。檔案以
單聲道載入、寫成 16 位元 PCM；輸出檔名已被佔用時會換一個，除非 `overwrite=True`。

## 兩個固定配方 {#two-fixed-recipes}

[`trad_enhance`][kudio.enhance.traditional.trad_enhance] 是取自 Loizou《Speech Enhancement:
Theory and Practice》的一個配方：512 樣本的 Hamming 音框、50 % 重疊、MCRA 雜訊估計、依 SNR
決定係數的功率譜相減，最後是 Wiener 式的增益，沒有任何參數可調；在這段片段上 SI-SDR 是
14.09 dB。短於一個音框的片段會被拒絕。
[`wavelet_low_pass_filter`][kudio.enhance.traditional.wavelet_low_pass_filter] 對 `db2` 分解的
細節係數，以峰值的 `thres` 倍做軟閾值，預設 0.8 時等於很重的低通。

`kudio enhance in.wav out.wav --method omlsa --noise mcra --set q=0.4` 和
`kudio compare in.wav --reference clean.wav --noises all` 在 shell 裡做同樣的事
（[命令列](../cli.md)）。

[^boll]: S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction", IEEE Trans. Acoust., Speech, Signal Process. 27(2), pp. 113-120, 1979, doi:10.1109/TASSP.1979.1163209.
[^berouti]: M. Berouti, R. Schwartz and J. Makhoul, "Enhancement of speech corrupted by acoustic noise", Proc. ICASSP 1979, vol. 4, pp. 208-211, doi:10.1109/ICASSP.1979.1170788.
[^kamath]: S. Kamath and P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise", Proc. ICASSP 2002, p. IV-4164, doi:10.1109/ICASSP.2002.5745591. The proceedings print the abstract only.
[^em84]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 32(6), pp. 1109-1121, 1984, doi:10.1109/TASSP.1984.1164453.
[^em85]: Y. Ephraim and D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator", IEEE Trans. Acoust., Speech, Signal Process. 33(2), pp. 443-445, 1985, doi:10.1109/TASSP.1985.1164550.
[^omlsa]: I. Cohen and B. Berdugo, "Speech enhancement for non-stationary noise environments", Signal Processing 81(11), pp. 2403-2418, 2001, doi:10.1016/S0165-1684(01)00128-1.
[^mcra]: I. Cohen and B. Berdugo, "Noise estimation by minima controlled recursive averaging for robust speech enhancement", IEEE Signal Process. Lett. 9(1), pp. 12-15, 2002, doi:10.1109/97.988717.
