# STFT 與特徵 {#stft-and-features}

模型從 kudio 拿到的是每個音框一列的矩陣：對數功率頻譜圖、MFCC 或堆疊的對數梅爾向量。
範例依序在同一段 16 kHz、兩秒的片段上執行：

```python
import numpy as np
import kudio

sr = 16000
y = (0.1 * np.random.default_rng(0).standard_normal(2 * sr)).astype(np.float32)
```

還在進來的音訊有 [`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream]
（[錄音與即時檢視](live-audio.md)），它的分框方式不同，單位是 dB。

## 列就是音框 {#rows-are-frames}

這裡的函式都傳回 `(frames, bins)`，時間在前；要給頻率在前的程式用時再轉置。上面的片段在
預設參數下：

| 呼叫 | 形狀 | 這段片段 | 數值 |
|---|---|---|---|
| `kudio.waveform_to_spectrogram(y)` | `(frames, n_fft // 2 + 1)` | `(126, 257)` | log10 功率，float32 |
| `kudio.STFT().analyse(y)` | `(frames, bins)` | `(126, 257)` | complex64 |
| `kudio.mfcc(y, sr)` | `(1, mel_frames, n_mfcc)` | `(1, 63, 40)` | float32 |
| `kudio.melspectrogram(y, sr=sr)` | `(mel_frames - n_frames + 1, n_mels * n_frames)` | `(59, 320)` | dB，float32 |
| `kudio.melspectrogram(y, sr=sr, n_frames=1)` | `(mel_frames, n_mels)` | `(63, 64)` | dB，float32 |
| `kudio.stack_context(spec, 2)` | `(frames, bins * 5)` | `(126, 1285)` | 邊緣補值，float32 |
| `kudio.frame_windows(spec, 64)` | `(ceil(frames / 64), 64, bins)` | `(2, 64, 257)` | 尾端補值，float32 |
| `kudio.spectrogram_to_waveform(y, spec)` | `(len(y),)` | `(32000,)` | float32 |

`frames` 是 `len(y) // 256 + 1`；`mel_frames` 是 [`mfcc`][kudio.core.feature.mfcc] 和
[`melspectrogram`][kudio.core.feature.melspectrogram] 以 512 樣本跳距算出的音框數。兩者都收
`hop_length=`。

## 用一個物件存放 STFT 設定 {#one-object-for-the-stft-settings}

[`STFT`][kudio.core.stft.STFT] 把 `n_fft`、`hop_length`、`win_length`、`window` 和取樣率
放在一起，頻譜圖和從它重建的波形就不會各用各的設定：

```python
stft = kudio.STFT(sr=16000, n_fft=512, hop_length=256)
spec = stft.forward(y)
print(stft.win_length, stft.window, stft.n_bins, stft.n_frames(len(y)))
print(stft.frame_times(3), spec.shape, spec.dtype)
```

```text
512 hamming 257 126
[0.    0.016 0.032] (126, 257) float32
```

窗函數預設是 Hamming，`win_length` 預設等於 `n_fft`。轉換是置中的：第 *t* 個音框以第
`t * hop_length` 個樣本為中心。它只收一個聲道；`(frames, channels)` 陣列會被
[`FeatureError`][kudio.exceptions.FeatureError] 拒絕。這個物件不可變。

## 對數功率，以及 `norm=True` 做了什麼 {#log-power-and-what-normtrue-does}

[`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] 傳回
`log10(|X|**2)`，單位是 bel；乘 10 就是 dB。複數頻譜會先加上 2.2e-16，所以數位靜音的
bin 是 -31.3；16 位元檔案的捨入雜訊大約在 -8。底部礙事的話，用 `np.maximum(spec, -10)`
截掉。

`norm=True` 用片段自己的音框把每個 bin 標準化成平均 0、標準差 1。這會拿掉音量和整體的
頻譜形狀，而且統計量不會傳回，所以這種頻譜圖轉不回音訊。輸出要變回聲音的模型，要用
[`Standardizer`][kudio.core.feature.Standardizer]（[下面](#standardising-with-the-training-sets-statistics)）。

## 梅爾頻譜圖與 MFCC {#mel-spectrograms-and-mfcc}

[`melspectrogram`][kudio.core.feature.melspectrogram] 傳回堆疊的對數梅爾向量：第 *i* 列是
第 *i* 到第 *i* + `n_frames` - 1 個音框，所以預設（`n_mels=64`、`n_frames=5`）每列 320 個值，
以第 *i* + 2 個音框為中心。

```python
feats = kudio.melspectrogram(y, sr=sr)
mel = kudio.melspectrogram(y, sr=sr, n_frames=1)
aligned = kudio.melspectrogram(y, sr=sr, n_frames=1, n_fft=512, hop_length=256)
coeffs = kudio.mfcc(y, sr, n_mfcc=13, n_fft=512, hop_length=256)[0]
print(feats.shape, mel.shape, aligned.shape, coeffs.shape)
```

```text
(59, 320) (63, 64) (126, 64) (126, 13)
```

![同一段片段畫成線性對數功率頻譜圖，以及 64 頻帶的梅爾頻譜圖](../images/features-linear-mel-light.webp#only-light)
![同一段片段畫成線性對數功率頻譜圖，以及 64 頻帶的梅爾頻譜圖](../images/features-linear-mel-dark.webp#only-dark)

1 kHz（虛線）以下，在線性圖裡佔八分之一，在梅爾圖裡佔三分之一。`source` 可以是路徑（以
單聲道載入）或陣列，陣列要用關鍵字給 `sr=`。預設是 `n_fft=1024`、`hop_length=512` 和
Hann 窗。短於 `n_frames` 個音框的片段會得到空的 `(0, n_mels * n_frames)` 陣列。

[`mfcc`][kudio.core.feature.mfcc] 收陣列和它的取樣率，傳回 `(1, mel_frames, n_mfcc)`；
`[0]` 拿掉批次軸。沒給時 `n_fft` 是 2048、`hop_length` 512、`n_mels` 128、窗函數 Hann。
`n_mfcc` 預設 40，不能超過 `n_mels`。

## 前後文與視窗 {#context-and-windows}

### `stack_context` {#stack_context}

[`stack_context`][kudio.core.feature.stack_context] 把每個音框和它前後各 `context` 個音框
接起來，最早的在前。邊緣會補值，所以輸出的第 *t* 列屬於第 *t* 個音框：

```python
frames = np.arange(12, dtype=np.float32).reshape(4, 3)
print(kudio.stack_context(frames, context=1))
```

```text
[[ 0.  1.  2.  0.  1.  2.  3.  4.  5.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.  8.]
 [ 3.  4.  5.  6.  7.  8.  9. 10. 11.]
 [ 6.  7.  8.  9. 10. 11.  9. 10. 11.]]
```

預設的 `pad="edge"` 重複第一和最後一個音框，因為全零的列在對數功率裡不是靜音；
`pad="zero"`、`forward_backward=` 和 `STFT.forward(context=)` 補零。

### `ContextFrames` {#contextframes}

[`ContextFrames`][kudio.core.feature.ContextFrames] 把一串特徵矩陣只存一份，要哪幾列才堆疊
哪幾列，每段語句在自己的邊緣補值，第 *i* 列就是下面 `np.vstack` 的第 *i* 列：

```python
rng = np.random.default_rng(0)
utterances = [rng.standard_normal((n, 257)).astype(np.float32) for n in (300, 120, 410)]
bank = kudio.ContextFrames(utterances, context=5)
full = np.vstack([kudio.stack_context(u, 5) for u in utterances])
print(bank.shape, np.array_equal(bank[[0, 5, 829]], full[[0, 5, 829]]))
```

```text
(830, 2827) True
```

### `frame_windows` {#frame_windows}

[`frame_windows`][kudio.core.feature.frame_windows] 把矩陣切成不重疊、每個 `n_frames` 長的
視窗，尾端用最後一個音框的複本補齊，`pad="zero"` 則補零：

```python
windows = kudio.frame_windows(spec, n_frames=64)
print(windows.shape, np.array_equal(windows.reshape(-1, 257)[:len(spec)], spec))
```

```text
(2, 64, 257) True
```

記下 `len(spec)`：損失函數要跳過補上的音框，轉回音訊時也要把它們切掉。沒有音框的矩陣會
丟出 `FeatureError`。

## 用訓練集的統計量標準化 {#standardising-with-the-training-sets-statistics}

[`Standardizer`][kudio.core.feature.Standardizer] 存著每個特徵維度的平均和標準差，在訓練集
的音框上擬合。這兩個陣列和權重一樣，是訓練好的模型的一部分。

```python
std = kudio.Standardizer().fit(spec)     # fit on the training set's frames
x = std.transform(windows)               # the model's input
print(x.shape, x.dtype, np.allclose(std.inverse(x), windows, atol=1e-4))
```

```text
(2, 64, 257) float32 True
```

`fit` 只收 2-D 矩陣，所以視窗的補值不會算進統計量；
[`inverse`][kudio.core.feature.Standardizer.inverse] 會還原 `transform`。標準差會加上
1e-8：請檢查 `(std.std < 1e-6).any()`。

### 把幾何設定和統計量一起存起來 {#saving-the-geometry-and-the-statistics-together}

[`to_dict`][kudio.core.stft.STFT.to_dict] 把 STFT 設定變成純 JSON，給 `STFT.from_dict` 讀；
`std.save("stats.npz")` 寫出統計量，給 `Standardizer.load` 讀；兩個都放在 checkpoint 旁邊。
`from_dict` 遇到不認得或缺少的鍵會拒絕，不會補上模型沒訓練過的預設值。轉換從不重新取樣；
[`matches`][kudio.core.stft.STFT.matches] 會說音訊要不要先 [`resample`][kudio.core.io.resample]。

## 轉回波形 {#back-to-a-waveform}

對數功率頻譜圖沒有相位，所以 [`STFT.inverse`][kudio.core.stft.STFT.inverse] 和
[`spectrogram_to_waveform`][kudio.core.feature.spectrogram_to_waveform] 從參考波形取相位，
通常是模型剛清理過的那段有雜訊的輸入：

```python
cleaned = std.inverse(x).reshape(-1, stft.n_bins)[:len(spec)]   # the prediction
cleaned[:, stft.n_bins // 2:] -= 2.0        # bins from 4 kHz up, 20 dB down
out = stft.inverse(y, cleaned)              # magnitude from cleaned, phase from y
change = stft.forward(out) - stft.forward(y)
print(out.shape, out.dtype, round(float(change[:, stft.n_bins // 2:].mean()), 3))
```

```text
(32000,) float32 -1.987
```

每一列都必須是同一組幾何設定下的 log10 功率，參考波形的每個音框一列；堆疊過前後文的列，
或還帶著補值的視窗，會丟出寫明兩邊形狀的 `FeatureError`。輸出剛好 `len(y)` 個樣本；沒改過
的頻譜圖來回一趟，誤差最多 6e-8。[`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave]
從檔案做同樣的事，寫成 16 位元 PCM。

要相位本身的話，[`analyse`][kudio.core.stft.STFT.analyse] 給出複數頻譜，
[`synthesise`][kudio.core.stft.STFT.synthesise] 把它轉回去；請傳 `length=len(y)`，不然會停在
最後一個完整的跳距。

## 一次處理很多檔案 {#many-files-at-once}

```python
from pathlib import Path

for name, seconds in [("a", 2.0), ("b", 1.5), ("c", 1.0)]:
    kudio.save_wave(f"data/clean/{name}.wav", y[:int(seconds * sr)], sr)
files = sorted(Path("data/clean").glob("*.wav"))
rows = kudio.logspec_from_files(files, sequence=False, forward_backward=0, norm=False)
seqs = kudio.logspec_from_files(files, sequence=True, forward_backward=0,
                                norm=False, desired_samples=sr)
coeffs = kudio.mfcc_from_files(files, n_mfcc=13, desired_samples=sr)
print(rows.shape, seqs.shape, coeffs.shape)
```

```text
(283, 257) (3, 63, 257) (3, 32, 13)
```

[`logspec_from_files`][kudio.core.feature.logspec_from_files] 和
[`mfcc_from_files`][kudio.core.feature.mfcc_from_files] 把各檔案的結果疊起來；3-D 的結果需要
每個檔案的音框數一樣，沒有檔案比 `desired_samples` 短時它就能做到。每個檔案以自己的取樣率
讀取。完整的參數在 [STFT 與特徵](../reference/features.md)。
