# 語音活動與音高 {#speech-activity-and-pitch}

[`vad`][kudio.core.vad.vad] 和它的相關函式找出哪裡有人在說話，只用 numpy。
[`f0`][kudio.core.pitch.f0] 用 pYIN 追蹤音高，並把有聲／無聲的判斷放在旁邊。

| 要做的事 | 用 |
|---|---|
| 找出哪裡有人說話，以秒為單位的 `(start, end)` | [`vad`][kudio.core.vad.vad] |
| 把每段語音切成獨立的片段 | [`vad_split`][kudio.core.vad.vad_split] |
| 去掉第一個字之前和最後一個字之後的部分 | [`vad_trim`][kudio.core.vad.vad_trim] |
| 找出資料夾裡沒有語音的錄音 | [`speech_ratio`][kudio.core.vad.speech_ratio] |
| 標出每個頻譜圖音框是不是語音 | [`vad_frames`][kudio.core.vad.vad_frames] |
| 追蹤音高，以及哪裡有音高 | [`f0`][kudio.core.pitch.f0]，傳回 [`PitchTrack`][kudio.core.pitch.PitchTrack] |
| 在編輯器裡打開結果 | [`Label`][kudio.core.dataset.Label]、[`save_labels`][kudio.core.dataset.save_labels] |

範例用下面這段產生的三個檔案。請在空資料夾裡執行。

??? example "本頁使用的錄音"

    三個音高滑動的有聲字、一個接進第二個字的 /s/，還有一陣不是語音的大聲寬頻碰撞聲：
    `quiet.wav` 在安靜的房間裡，`fan.wav` 在風扇前面，`empty.wav` 只有碰撞聲和風扇，
    沒有人說話。

    ```python
    import numpy as np
    import kudio

    sr = 16000
    n = 6 * sr
    rng = np.random.default_rng(7)

    def at(start, x):                      # place x at `start` seconds
        out = np.zeros(n)
        out[int(start * sr):int(start * sr) + len(x)] = x
        return out

    def vowel(seconds, f_from, f_to, level=1.0):   # harmonics of a gliding f0
        m = int(seconds * sr)
        phase = 2 * np.pi * np.cumsum(np.linspace(f_from, f_to, m)) / sr
        tone = sum(np.sin(k * phase) / k for k in range(1, 16))
        return 0.08 * level * tone * np.sin(np.pi * np.arange(m) / m) ** 0.5

    def hiss(seconds, level, cutoff=None):         # a noise burst
        m = int(seconds * sr)
        x = rng.standard_normal(m)
        if cutoff:
            x = kudio.highpass(x, sr, cutoff=cutoff)
        return level * x * np.hanning(m)

    voice = (at(0.8, vowel(0.8, 140, 110))
             + at(3.0, hiss(0.15, 0.05, cutoff=3500))      # an /s/ ...
             + at(3.12, vowel(0.85, 120, 165))             # ... running into a vowel
             + at(4.6, vowel(0.8, 130, 100, level=0.5)))   # a quieter last word
    clatter = at(2.2, hiss(0.25, 0.25))                    # loud, broadband, not speech
    room = 0.0005 * rng.standard_normal(n)                 # a quiet room
    fan = 0.03 * kudio.lowpass(rng.standard_normal(n), sr, cutoff=1000)

    kudio.save_wave("quiet.wav", voice + clatter + room, sr)
    kudio.save_wave("fan.wav", voice + clatter + fan, sr)
    kudio.save_wave("empty.wav", clatter + fan, sr)       # nobody talking
    ```

## 哪裡有人在說話 {#where-somebody-is-talking}

```python
for name in ("quiet.wav", "fan.wav"):
    y, sr = kudio.file_load(name)
    spans = kudio.vad(y, sr)
    parts = kudio.split_on_silence(y)
    print(f"{name:9}", [(round(a, 2), round(b, 2)) for a, b in spans], len(parts), "parts")
```

```text
quiet.wav [(0.73, 1.67), (2.94, 4.04), (4.53, 5.47)] 4 parts
fan.wav   [(0.77, 1.64), (2.99, 4.01), (4.64, 5.36)] 1 parts
```

[`vad`][kudio.core.vad.vad] 傳回以秒為單位的 `(start, end)`，找不到像語音的東西時傳回空
清單；兩個房間都不用改任何設定，就找到三個字。像
[`split_on_silence`][kudio.effects.silence.split_on_silence] 這種只看位準的門檻，在安靜的
房間找到四段，碰撞聲也算在內；在風扇前面只找到一段六秒長的，因為風扇的位準離最響的音框
不到 30 dB。它適合安靜房間裡不是語音的聲音。

![兩個檔案的音框位準，畫出 top_db 線和 vad 找到的區段](../images/speech-level-light.webp#only-light)
![兩個檔案的音框位準，畫出 top_db 線和 vad 找到的區段](../images/speech-level-dark.webp#only-dark)

## 切割、裁切與計算比例 {#cutting-trimming-and-counting}

```python
y, sr = kudio.file_load("fan.wav")
for n, part in enumerate(kudio.vad_split(y, sr), 1):
    kudio.save_wave(f"spans/fan_{n:03d}.wav", part, sr)
clip, (start, end) = kudio.vad_trim(y, sr)
print(start, end, round(len(clip) / sr, 2))
for name in ("empty.wav", "fan.wav", "quiet.wav"):
    print(f"{name:10} {kudio.speech_ratio(kudio.file_load(name)[0], sr):4.0%}")
```

```text
12320 85760 4.59
empty.wav    0%
fan.wav     43%
quiet.wav   50%
```

[`vad_split`][kudio.core.vad.vad_split] 把各區段變成波形傳回，[`vad_trim`][kudio.core.vad.vad_trim]
保留第一段到最後一段語音，傳回 `(clip, (start, end))`，單位是樣本；
[`speech_ratio`][kudio.core.vad.speech_ratio] 給出語音所佔的比例。找不到語音時，`vad_trim`
原封不動傳回片段，並在 INFO 記一筆。`empty.wav` 的碰撞聲比任何一個字都響，讀數仍是 0 %。
`vad` 的底噪取自檔案本身，所以要對整段錄音用：已經切到只剩字的片段，自己就是底噪。
`kudio vad fan.wav --split-to spans/` 在 shell 裡做同樣的事，沒有語音時結束碼是 1。

## 每個頻譜圖音框一個判斷 {#one-decision-per-spectrogram-frame}

```python
spec = kudio.waveform_to_spectrogram(y)        # (frames, bins), hop 256
speech = kudio.vad_frames(y, sr)               # (frames,), the same frames
print(spec.shape, speech.shape, speech.sum())
background = np.median(spec[~speech], axis=0)  # per bin, over what vad turned down
```

```text
(376, 257) (376,) 144
```

[`vad_frames`][kudio.core.vad.vad_frames] 每個 STFT 音框給一個布林值，音框中心落在區段內就是
`True`，音框和預設參數下的 [`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram]
一致；改了 `hop_length`、`n_fft` 或 `center` 時，這裡也要給同樣的值。`~speech` 也包含碰撞聲，
所以範例取中位數。

## vad 如何判斷 {#how-vad-decides}

[`vad`][kudio.core.vad.vad] 每 10 ms 看一個 30 ms 的音框。音框要比底噪（音框位準的第 15
百分位）高 [`THRESHOLD_DB`][kudio.core.vad.THRESHOLD_DB]（8 dB），而且比最響的音框低不到
[`FLOOR_REL_DB`][kudio.core.vad.FLOOR_REL_DB]（45 dB）。接著要像人聲：頻譜平坦度低於
[`FLATNESS_MAX`][kudio.core.vad.FLATNESS_MAX]（0.40；母音低於 0.001，30 ms 的白雜訊大約
0.56），或過零率高於 [`ZCR_MIN`][kudio.core.vad.ZCR_MIN]（0.25）且在通過的音框
[`FRICATIVE_REACH`][kudio.core.vad.FRICATIVE_REACH]（0.2 s）以內，這是給擦音的。不超過
`min_silence` 的空隙會被接起來，短於 `min_speech` 的區段會丟掉，兩端各加上 `pad`。

| 參數 | 預設 | 什麼時候改 |
|---|---|---|
| `threshold_db` | 8.0 dB | 小聲的人被漏掉（調低），或吵的房間被當成語音（調高） |
| `flatness_max` | 0.40 | 音訊不是語音（1.0 會關掉這項檢查） |
| `min_speech` | 0.10 s | 喀聲和敲擊聲變成短區段（調高） |
| `min_silence` | 0.15 s | 字在塞音處被切開（調高），或不同的句子被接在一起（調低） |
| `pad` | 0.05 s | 切掉了第一個或最後一個子音（調高） |
| `band` | `None`，整個頻帶 | 高取樣率的錄音在語音頻帶之上有很強的底噪：`(100, 4000)` |

會出錯的地方：從檔案中途開始、蓋不到大約 85 % 的雜訊會被當成語音，所以要在背景改變的地方
把檔案切開；只佔部分頻譜的雜訊，例如 500 Hz 以下的低頻隆隆聲或 3.5 kHz 以上的嘶聲，會通過
平坦度檢查；緊貼著字的寬頻雜訊會併進那個字；片段裡要有 15 % 的音框比語音安靜。多聲道陣列
會丟出 [`FeatureError`][kudio.exceptions.FeatureError]。

## 用 f0 追蹤音高 {#pitch-with-f0}

```python
y, sr = kudio.file_load("fan.wav")
track = kudio.f0(y, sr)
print(track)
print(len(track), track.hop_length, track.times[75:80])
print(track.f0[75:80].round(1))
```

```text
median 125.6 Hz  ·  103.8-158.2 Hz (7.3 semitones)  ·  45% voiced
601 160 [0.75 0.76 0.77 0.78 0.79]
[  nan   nan 140.1 140.1 139.3]
```

[`f0`][kudio.core.pitch.f0] 執行 pYIN[^pyin]，傳回 [`PitchTrack`][kudio.core.pitch.PitchTrack]：
`track.f0` 單位是 Hz，無聲的音框是 `NaN`；`track.voiced` 是判斷本身，由整段片段上的隱藏式
馬可夫模型決定，所以在一個音節裡不會跳來跳去。跳距預設 10 ms，可用 `hop_length` 改。一分鐘的
16 kHz 音訊大約要 0.8 s，只想知道有沒有人聲的話，[`speech_ratio`][kudio.core.vad.speech_ratio]
快得多。

![fan.wav 的頻譜圖，上面畫著 f0 軌跡](../images/speech-pitch-light.webp#only-light)
![fan.wav 的頻譜圖，上面畫著 f0 軌跡](../images/speech-pitch-dark.webp#only-dark)

搜尋範圍是 65 到 400 Hz（[`SPEECH_FMIN`][kudio.core.pitch.SPEECH_FMIN] 和 `SPEECH_FMAX`），
範圍外的音高會被回報成範圍內的某個值：

```python
t = np.arange(sr) / sr
tone = 0.3 * sum(np.sin(2 * np.pi * 500 * k * t) / k for k in range(1, 10))
print(kudio.f0(tone, sr))                  # the speech range, 65-400 Hz
print(kudio.f0(tone, sr, fmax=1000))       # a range that holds 500 Hz
```

```text
median 249.7 Hz  ·  249.7-249.7 Hz (0.0 semitones)  ·  100% voiced
median 499.4 Hz  ·  499.4-499.4 Hz (0.0 semitones)  ·  100% voiced
```

第一行低了一個八度；中位數剛好落在 `fmin` 也是要留意的徵兆。唱歌或兒童的聲音請自己設
`fmin` 和 `fmax`。`fmin` 不在 0 和 `fmax` 之間、`fmax` 超過 Nyquist、`frame_length` 短於
`2 * floor(sr / fmin) + 2` 時會丟出 `FeatureError`。沒給的話 `frame_length` 是 2048，
88.2 和 96 kHz 時是 4096。

`print(track)` 顯示中位數和第 5 到第 95 百分位的範圍，一個八度錯誤動不了它們；
[`summary()`][kudio.core.pitch.PitchTrack.summary] 以 dict 傳回同樣的數字。pYIN 的有聲判斷
不看位準，所以只要語音裡面的音高時，用軌跡的跳距配上 `vad_frames`：
`track.f0[track.voiced & kudio.vad_frames(y, sr, hop_length=track.hop_length)]`。

## 標籤：在編輯器裡打開結果 {#labels-opening-the-result-in-an-editor}

```python
speech = [kudio.Label(a, b, f"speech {n}")
          for n, (a, b) in enumerate(kudio.vad(y, sr), 1)]
kudio.save_labels("fan.txt", speech)
kudio.save_labels("fan.voiced.txt", track.to_labels())
print(open("fan.voiced.txt").read())
```

```text
0.770000	1.650000	voiced 1
3.080000	3.990000	voiced 2
4.570000	5.460000	voiced 3
```

[`save_labels`][kudio.core.dataset.save_labels] 寫出 Audacity 的標籤軌；在 Audacity 裡打開
`fan.wav`，再用 *File › Import › Labels* 加入這些檔案。
[`to_labels()`][kudio.core.pitch.PitchTrack.to_labels] 會略過短於 0.05 s 的有聲段落
（`min_seconds`）。把修正過的標籤讀回來，見[編輯與效果](editing.md)。
`kudio pitch fan.wav --segments --labels fan.voiced.txt` 印出摘要和有聲段落、寫出標籤軌，
沒有有聲段落時結束碼是 1。

[^pyin]: M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", Proc. ICASSP 2014, pp. 659-663, doi:10.1109/ICASSP.2014.6853678.
