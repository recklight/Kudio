# 編輯與效果 {#editing-and-effects}

`kudio.effects` 把音訊編輯器對波形做的事寫成函式：淡入淡出、濾波、剪接、修剪、音量，以及
訓練用的資料擴增。每個函式都收 NumPy 陣列、傳回新的陣列，不會改寫你傳進來的那個。通道劣化
有自己的一頁：[房間與通道](degradation.md)。

| 要做的事 | 用 |
|---|---|
| 不讓片段在開頭或結尾發出喀聲 | [`fade`][kudio.effects.edit.fade] |
| 去掉低頻隆隆聲、嗡聲，或某個頻帶以外的一切 | [`highpass`][kudio.effects.filters.highpass]、[`lowpass`][kudio.effects.filters.lowpass]、[`bandpass`][kudio.effects.filters.bandpass]、[`bandstop`][kudio.effects.filters.bandstop] |
| 去掉固定的偏移 | [`remove_dc`][kudio.effects.edit.remove_dc] |
| 剪掉一段，或補進一段 | [`splice`][kudio.effects.edit.splice] |
| 去掉錄音室錄音前後的靜音 | [`trim_silence`][kudio.effects.silence.trim_silence]、[`split_on_silence`][kudio.effects.silence.split_on_silence] |
| 依標籤檔切割錄音 | [`load_labels`][kudio.core.dataset.load_labels] |
| 設定峰值 | [`normalize`][kudio.effects.augment.normalize]、[`normalize_db`][kudio.effects.augment.normalize_db] |
| 從一段片段得到更多訓練資料 | [`pitch_shift`][kudio.effects.augment.pitch_shift]、[`time_stretch`][kudio.effects.augment.time_stretch]、[`add_noise_snr`][kudio.effects.augment.add_noise_snr]、[`reverb`][kudio.effects.augment.reverb]、[`spec_augment`][kudio.effects.augment.spec_augment] |

??? example "這一頁用到的檔案"

    `take.wav` 是六秒、RMS 0.01 的白雜訊，代表房間，裡面有兩段有聲的語句：1.0 到 2.0 s 和
    3.5 到 4.8 s。`retake.wav` 是要補進去的 0.3 s 語句，`stereo.wav` 是放在兩個聲道上的
    take，`take.txt` 是標出這兩段語句的 Audacity 標籤軌。

    ```python
    import numpy as np
    import kudio

    sr = 16000

    def voiced(seconds, f0):
        t = np.arange(int(seconds * sr)) / sr
        return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) / 6

    y = 0.01 * np.random.default_rng(0).standard_normal(6 * sr)    # the room
    y[1 * sr:2 * sr] += voiced(1.0, 120)
    y[int(3.5 * sr):int(4.8 * sr)] += voiced(1.3, 150)
    kudio.save_wave("take.wav", y.astype(np.float32), sr)
    kudio.save_wave("retake.wav", (0.5 * voiced(0.3, 130)).astype(np.float32), sr)
    kudio.save_wave("stereo.wav", np.stack([y, 0.5 * y], axis=1).astype(np.float32), sr)
    kudio.save_labels("take.txt", [kudio.Label(0.93, 2.07, "speech"),
                                   kudio.Label(3.43, 4.87, "speech")])
    ```

## 聲道與取樣率 {#channels-and-sample-rates}

以秒為單位的函式都用 `sr` 收取樣率。這裡每個函式都收單聲道的 `(n,)` 或 `(frames, channels)`。
淡入淡出、`remove_dc`、`splice`、`crossfade` 和音量函式對每個聲道做同樣的事，所有聲道同一個
增益；濾波器、`pitch_shift`、`time_stretch` 和 `reverb` 沿著 frame 各自處理每個聲道；靜音
函式在同樣的樣本位置切所有聲道。[`add_noise_snr`][kudio.effects.augment.add_noise_snr] 只收
單聲道。`splice` 和 `crossfade` 不檢查各段的取樣率是否相同，所以每段都要用同一個取樣率載入。

## 淡入淡出與直流偏移 {#fades-and-dc-offset}

```python
y, sr = kudio.file_load("take.wav")
y = kudio.fade(y, sr, fade_in_s=0.01, fade_out_s=0.01)
```

從長錄音裡切出來的片段，開頭和結尾都停在波形中間，會發出喀聲；兩端各 10 ms 就能去掉，又
不會聽成淡入淡出。[`fade_in`][kudio.effects.edit.fade_in] 和 [`fade_out`][kudio.effects.edit.fade_out]
各處理一端。`shape=` 可以是 `'linear'`（預設，用來防喀聲）、`'cosine'`（兩端平坦，用在聽得到的
淡入淡出）或 `'exponential'`（從 -60 dB 起在 dB 上是直線，用在慢慢淡出）。淡入淡出比片段長，
或淡入和淡出重疊時，會丟出 [`FeatureError`][kudio.exceptions.FeatureError]。

[`remove_dc`][kudio.effects.edit.remove_dc] 減去每個聲道的平均值。固定的偏移會吃掉餘裕，
也會讓 `trim_silence` 修不掉任何東西；會漂移的偏移其實是極低頻，要用 `highpass` 去掉。

## 濾波器 {#filters}

```python
y = kudio.highpass(y, sr, cutoff=80)       # rumble, handling noise
y = kudio.bandstop(y, sr, 45, 55)          # 50 Hz hum; (55, 65) for 60 Hz
y = kudio.lowpass(y, sr, cutoff=7000)      # hiss
```

| Butterworth，4 階（`order=`） | 預設 | 用途 |
|---|---|---|
| [`highpass`][kudio.effects.filters.highpass] | 去掉 80 Hz 以下 | 低頻隆隆聲、手持雜音、慢慢漂移 |
| [`lowpass`][kudio.effects.filters.lowpass] | 去掉 8000 Hz 以上 | 嘶聲，以及 16 kHz 模型頻帶以上的一切 |
| [`bandpass`][kudio.effects.filters.bandpass] | 保留 300–3400 Hz | 電話頻帶 |
| [`bandstop`][kudio.effects.filters.bandstop] | 去掉 45–55 Hz | 50 Hz 市電嗡聲 |

`lowpass` 的截止頻率在 Nyquist 或以上時沒有東西可去，傳回一份複本；其他邊緣必須嚴格落在 0
和 Nyquist 之間，頻帶邊緣反過來給會被拒絕。[`band_filter`][kudio.effects.filters.band_filter]
用 `btype=` 指定種類，給執行時才決定濾波器的程式用。

濾波器先順向再逆向各跑一次（`sosfiltfilt`），所以不會有時間位移，輸出和輸入逐樣本對齊。
這會把響應平方，所以截止頻率上的純音會低 6 dB，而不是 3 dB：

```python
import numpy as np

tone = np.sin(2 * np.pi * 1000 * np.arange(2 * sr) / sr)     # at the cutoff
middle = slice(sr // 2, 3 * sr // 2)                         # away from the ends
for zero_phase in (True, False):
    out = kudio.lowpass(tone, sr, cutoff=1000, zero_phase=zero_phase)
    gain_db = 20 * np.log10(np.std(out[middle]) / np.std(tone[middle]))
    print(f"zero_phase={zero_phase}: {gain_db:.2f} dB")
```

```text
zero_phase=True: -6.02 dB
zero_phase=False: -3.01 dB
```

`zero_phase=False` 是只跑一次的因果濾波器，和即時運作的裝置一樣，延遲隨頻率不同。

## 剪下與接合 {#cutting-and-joining}

```python
y, sr = kudio.file_load("take.wav")
retake, _ = kudio.file_load("retake.wav", sr=sr)    # at the take's rate
a, b = int(1.20 * sr), int(1.45 * sr)               # a cough, say
fixed = kudio.splice([y[:a], y[b:]], sr)                 # cut it out
patched = kudio.splice([y[:a], retake, y[b:]], sr)       # or replace it
print(len(y) - (b - a) - len(fixed))
```

```text
80
```

[`splice`][kudio.effects.edit.splice] 把每段的最後 5 ms 和下一段的前 5 ms 混在一起
（[`DEFAULT_CROSSFADE`][kudio.effects.edit.DEFAULT_CROSSFADE]），`np.concatenate` 則會留下
一個會發出喀聲的跳動。結果比各段加起來短，每個接縫短一個重疊的長度，所以跟著音訊對齊的標籤
也要跟著移。空的段落會略過，不同型別的段落各按自己的刻度讀。

![用兩種方式接合刪掉一段之後的錄音：np.concatenate 留下一個樣本的跳動和頻譜圖上的寬頻條紋；splice 沒有跳動，條紋弱 44 dB](../images/editing-seam-light.webp#only-light)
![用兩種方式接合刪掉一段之後的錄音：np.concatenate 留下一個樣本的跳動和頻譜圖上的寬頻條紋；splice 沒有跳動，條紋弱 44 dB](../images/editing-seam-dark.webp#only-dark)

[`crossfade`][kudio.effects.edit.crossfade] 用一個接縫接起兩段片段，預設 `shape='equal_power'`，
`splice` 則預設 `'linear'`。線性交叉淡化讓延續下去的同一個訊號音量不變；等功率讓兩段不相干的
錄音音量不變，但用在同一個訊號上最多會大 3 dB。

## 依靜音修剪與切割 {#trimming-and-splitting-on-silence}

```python
tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)
y = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)]).astype(np.float32)
for clip in (y, y + 0.02, kudio.remove_dc(y + 0.02)):
    _, (start, end) = kudio.trim_silence(clip, top_db=30)
    print(start / sr, end / sr)
```

```text
0.448 1.088
0.0 1.5
0.448 1.088
```

[`trim_silence`][kudio.effects.silence.trim_silence] 修掉兩端的靜音，傳回片段和保留的範圍；
[`split_on_silence`][kudio.effects.silence.split_on_silence] 在每個空隙切開。RMS 比最響的
音框低 `top_db`（30）以上的音框算靜音；音框 2048 個樣本、跳距 512，切點往外靠。0.02 的直流
偏移讓每個音框都在門檻之上，所以要等 `remove_dc` 去掉它才修得掉。只看位準在 `take.wav` 這種
有房間底噪的背景上行不通；語音請用 [`vad`][kudio.core.vad.vad]（[語音活動與音高](speech.md)）。
`kudio trim` 在 shell 裡做修剪。

## 依標籤切割 {#cutting-at-labels}

```python
y, sr = kudio.file_load("take.wav")
for label in kudio.load_labels("take.txt"):
    clip = y[round(label.start * sr):round(label.end * sr)]
    print(label.text, round(label.duration, 2), len(clip))
```

```text
speech 1.14 18240
speech 1.44 23040
```

[`load_labels`][kudio.core.dataset.load_labels] 把 Audacity 標籤軌或 JSON 讀回成以秒為單位
的 [`Label`][kudio.core.dataset.Label]；轉成樣本要用 `round`，不要用 `int`，`int` 會讓 2.01 s
少一個樣本。Audacity 的頻率列會被略過；不是標籤的列、不是有限值的時間、結束早於開始，都是
寫明行號的 `ValueError`。

## 音量 {#level}

[`normalize`][kudio.effects.augment.normalize] 和 [`normalize_db`][kudio.effects.augment.normalize_db]
設定峰值，以滿刻度的比例或 dBFS 表示，靜音的片段原樣保留。[`gain`][kudio.effects.augment.gain]
和 [`random_gain`][kudio.effects.augment.random_gain] 以固定或依種子隨機的 dB 數改變音量。
整數音訊保持原本的型別，按自己的滿刻度換算，超過時削峰而不是繞回。要讓片段聽起來一樣大聲，
請用 [`normalize_lufs`][kudio.core.loudness.normalize_lufs]（[檢查錄音](inspection.md)）。

## 資料擴增 {#augmentation}

有隨機成分的都收 `seed`，同一個種子給出同一個陣列。

[`pitch_shift`][kudio.effects.augment.pitch_shift] 把音高移 `n_steps` 個半音，長度不變；
[`time_stretch`][kudio.effects.augment.time_stretch] 以 `rate` 改變長度（大於 1 是變快），
音高不變。兩者都是相位以 float64 計算的相位聲碼器。

```python
y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
noise = np.random.default_rng(0).standard_normal(10 * sr).astype(np.float32)
mix = kudio.add_noise_snr(y, noise, snr_db=5, seed=0)
print(round(kudio.snr(y, mix), 2), np.array_equal(mix, kudio.add_noise_snr(y, noise, 5, seed=0)))
```

```text
5.0 True
```

[`add_noise_snr`][kudio.effects.augment.add_noise_snr] 以 `snr_db` 把單聲道的 `noise` 混進
`y`，SNR 以整段片段計算，停頓也算在內。較短的雜訊會重複鋪滿，較長的雜訊從依種子決定的位置
裁切；靜音或固定值的雜訊會丟出 `FeatureError`。整個語料庫請交給
[`Synthesizer`][kudio.core.synth.Synthesizer]（[建立含雜訊資料集](synthesis.md)）。

[`reverb`][kudio.effects.augment.reverb] 加上七個相隔 `delay_ms`（50 ms）的回音，每個是前一個
的 `decay`（0.3）倍，長度和峰值不變。聽起來是回音，不是房間；要房間請用 [`rir`][kudio.core.room.rir]
和 [`apply_rir`][kudio.core.room.apply_rir]（[房間與通道](degradation.md)）。

[`spec_augment`][kudio.effects.augment.spec_augment] 在 `(frames, bins)` 頻譜圖上隨機遮掉
頻帶：最多 `freq_mask_width`（8）個 bin 和 `time_mask_width`（16）個音框，預設各一條，寬度
從 0 起隨機抽。被遮的格子填頻譜圖的平均值，相當於論文裡先正規化再填零[^specaugment]；
`mask_value=` 可以改。參數在 [編輯與效果](../reference/effects.md)。

[^specaugment]: D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk
    and Q. V. Le, "SpecAugment: A simple data augmentation method for
    automatic speech recognition", Proc. Interspeech 2019, pp. 2613-2617,
    doi:10.21437/Interspeech.2019-2680, Sec. 2.
