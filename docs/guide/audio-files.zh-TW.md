# 音訊檔案 {#audio-files}

kudio 載入的波形都是 float32 的 NumPy 陣列，±1.0 是滿刻度：單聲道是 1-D，多聲道是
`(frames, channels)`。檔案透過 [soundfile](https://python-soundfile.readthedocs.io/)
讀寫，[soundfile 不讀的格式](#formats-soundfile-does-not-read)交給 ffmpeg。範例讀的是
下面這段產生的檔案；請在空資料夾裡執行。

??? example "產生範例要讀的檔案"

    ```python
    from pathlib import Path

    import numpy as np

    import kudio


    def tone(freq, seconds, sr=16000, amp=0.3):
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


    kudio.save_wave("clip.wav", tone(220, 3.5, amp=0.45), 16000)
    kudio.save_wave("stereo.wav", np.stack([tone(220, 1), tone(880, 1)], axis=1), 16000)
    for name, amp in [("spk1/a", 0.05), ("spk1/b", 0.2), ("spk2/a", 0.6)]:
        y = tone(330, 2.0, 44100, amp)
        kudio.save_wave(f"raw/{name}.wav", np.stack([y, 0.8 * y], axis=1), 44100)
    for name in ["one", "two"]:
        kudio.save_wave(f"incoming/{name}.wav", tone(440, 1.0), 16000)
    Path("incoming/broken.wav").write_bytes(b"this is not a wave file")
    for name in ["rec.wav", "take.wav", "take.v2.wav"]:
        kudio.save_wave(f"takes/{name}", tone(250, 0.5), 16000)
    ```

## 載入之前先看 {#looking-before-loading}

```python
info = kudio.audio_info("clip.wav")
print(info)
print(info.frames, info.duration, info.subtype, info.format)
```

```text
clip.wav: 3.500s, 16000 Hz, 1ch, PCM_16, peak 0.4500
56000 3.5 PCM_16 WAV
```

[`audio_info`][kudio.core.io.audio_info] 除了峰值以外都從檔頭讀。峰值每次讀 2^20 個
frame；`peak=False` 就不讀。也可以傳陣列，同時給 `sr=`。在 shell 裡 `kudio info`
印出一樣的內容。

## 載入檔案 {#loading-a-file}

```python
y, sr = kudio.file_load("clip.wav")                       # the file's own rate
y8, sr8 = kudio.file_load("clip.wav", sr=8000)            # resampled with soxr_hq
st, _ = kudio.file_load("stereo.wav", mono=False)         # channels kept
print(y.shape, sr, y8.shape, sr8, st.shape)
```

```text
(56000,) 16000 (28000,) 8000 (16000, 2)
```

[`file_load`][kudio.core.io.file_load] 讀得了 libsndfile 開得了的格式，包括 WAV、FLAC、
OGG、MP3、AIFF 和 NIST SPHERE。`mono=True` 是預設，會把各聲道平均。讀不了的檔案會丟出
[`AudioIOError`][kudio.exceptions.AudioIOError]，寫明路徑和原因；取樣率是 0、NaN、`True`
或 16000.5 則是 `ValueError`。

!!! warning "聲道在後"
    `librosa.load(..., mono=False)` 傳回 `(channels, frames)`。在 kudio 裡左聲道是
    `y[:, 0]`。兩列以上、而且欄數多於列數的陣列會被拒絕，`ValueError` 會叫你轉置。

### soundfile 不讀的格式 {#formats-soundfile-does-not-read}

m4a、mp4、aac、webm、3gp、amr 和 wma 在 kudio 讀檔的每個地方都用
[audioread](https://github.com/beetbox/audioread) 和 [ffmpeg](https://ffmpeg.org/) 解碼。
裝上這個套件組，再把 ffmpeg 放到 PATH 上；缺任何一個，`AudioIOError` 會說缺的是哪個：

```bash
pip install "kudio[compressed]"
```

編碼器會補滿最後一個區塊，所以 2 s 的聲音，從裸的 AAC 串流讀回來是 2.043 s，從 m4a
是 2.000 s：逐樣本比較之前，先[對齊這類檔案](metrics.md#finding-the-delay)。

### 截斷的檔案 {#a-file-that-was-cut-short}

資料比檔頭說的短的 WAV 會讀到哪算到哪，並發出警告，例如 `take.wav: the header says
6.00 s and 3.00 s are there, 96022 bytes fewer. The file may have been cut short.`
長度是 0 或 0x7FFFF000 以上（串流寫入程式會留下這種值）不算問題。截斷的 FLAC 會丟出例外。

## 寫入檔案 {#writing-a-file}

```python
import numpy as np

hot = np.array([1.2, -1.3, 0.5], dtype=np.float32)
kudio.save_wave("out/hot16.wav", hot, 16000)                  # 16-bit PCM
kudio.save_wave("out/hotf.wav", hot, 16000, subtype="FLOAT")  # 32-bit float
kudio.save_wave("out/hot.flac", hot, 16000)                   # FLAC, by extension
print(kudio.file_load("out/hot16.wav")[0])
print(kudio.file_load("out/hotf.wav")[0])
```

```text
[ 0.9999695 -1.         0.5      ]
[ 1.2 -1.3  0.5]
```

| `subtype` | 存成 | 超過 ±1.0 |
|---|---|---|
| `PCM_16`（預設） | 整數，65,536 階 | 削到滿刻度 |
| `PCM_24` | 整數，16,777,216 階 | 削到滿刻度 |
| `FLOAT` | 32 位元浮點數 | 照寫 |

[`save_wave`][kudio.core.io.save_wave] 會建立缺少的資料夾；新檔寫完整之後才取代舊檔；
`d_sample=` 會先重新取樣。subtype 要配得上容器：FLAC 收 `PCM_16` 和 `PCM_24`，不收
`FLOAT`。每種整數 subtype 都由 kudio 自己削峰和四捨五入；libsndfile 自己的轉換是無條件
捨去，平均低半個 16 位元階，關掉削峰時還會把 1.2 繞成 -0.8：

![左：峰值 1.25 的 220 Hz 純音，分別從 save_wave 寫的 FLOAT 檔（完整）、save_wave 寫的 PCM_16 檔（在正負 1 被削平）、以及關掉 libsndfile 削峰功能寫出的 PCM_16 檔（波峰繞到另一個正負號）讀回來。右：20,000 個隨機樣本寫成 16 位元的捨入誤差，經 soundfile.write 分布在 -1 到 0 階，經 save_wave 分布在 -0.5 到 +0.5 階。](../images/audio-files-full-scale-light.webp#only-light)
![左：峰值 1.25 的 220 Hz 純音，分別從 save_wave 寫的 FLOAT 檔（完整）、save_wave 寫的 PCM_16 檔（在正負 1 被削平）、以及關掉 libsndfile 削峰功能寫出的 PCM_16 檔（波峰繞到另一個正負號）讀回來。右：20,000 個隨機樣本寫成 16 位元的捨入誤差，經 soundfile.write 分布在 -1 到 0 階，經 save_wave 分布在 -0.5 到 +0.5 階。](../images/audio-files-full-scale-dark.webp#only-dark)

整數陣列在 kudio 裡一律按它自己型別的滿刻度換算：int16 乘 1/32768，int32 乘 1/2^31，
uint8 以 128 為中點。int64 會被拒絕，因為在它的刻度上 16 位元的樣本落在 288 dB 以下。

### NaN 和 inf {#nan-and-inf}

含 NaN 或 inf 的檔案以原取樣率載入時會發出警告並算出個數；換成其他取樣率則是
`AudioIOError`，因為重新取樣會把它們擴散出去。`save_wave` 拒絕這種樣本，什麼都不寫。

## 在記憶體中改變取樣率 {#changing-the-rate-in-memory}

```python
y8 = kudio.resample(y, orig_sr=sr, target_sr=8000)
print(len(y), len(y8), kudio.resample(y, orig_sr=sr, target_sr=sr) is y)
```

```text
56000 28000 True
```

[`resample`][kudio.core.io.resample] 用的是 `soxr_hq`，和 `file_load(path, sr=)` 一樣；
取樣率相同時傳回原陣列本身。`(frames, channels)` 陣列沿著 frame 重新取樣；整數陣列傳回
原本的型別。

## 蒐集檔案 {#gathering-files}

```python
files, rejected = kudio.check_input(["raw/", "takes/rec.wav", "missing.wav"])
print([f.as_posix() for f in files], rejected)
waves, sr, length = kudio.load_waves("raw/", std_len=True)
copied = kudio.copy_waves("picked/", "raw/")       # destination first
print(len(waves), sr, length, [p.name for p in copied])
```

```text
['raw/spk1/a.wav', 'raw/spk1/b.wav', 'raw/spk2/a.wav', 'takes/rec.wav'] ['missing.wav']
3 44100 88200 ['a.wav', 'b.wav', 'a2.wav']
```

[`check_input`][kudio.core.io.check_input] 是所有資料夾工具讀輸入的方式：檔案、任意深度
的資料夾、`.txt` 清單（UTF-8，相對路徑以工作目錄為準），以及這些東西組成的清單；傳回
`.wav` 檔和用不了的項目。[`load_waves`][kudio.core.io.load_waves] 以原取樣率載入單聲道，
`std_len=True` 會裁到最短；[`copy_waves`][kudio.core.io.copy_waves] 依檔名平鋪複製，
所以第二個 `a.wav` 會變成 `a2.wav`。

## 轉換整個資料夾 {#converting-a-whole-folder}

```python
result = kudio.convert_folder("raw/", "16k/", sr=16000, lufs=-23.0)
print(result)
for path in result.outputs:
    y, sr = kudio.file_load(path)
    print(path.as_posix(), sr, f"{kudio.loudness(y, sr):.1f} LUFS")
```

```text
3/3 written, 0 failed
16k/spk1/a.wav 16000 -23.0 LUFS
16k/spk1/b.wav 16000 -23.0 LUFS
16k/spk2/a.wav 16000 -23.0 LUFS
```

[`convert_folder`][kudio.core.io.convert_folder] 對每個檔案：載入（有 `sr=` 就用它，除非
`mono=False` 否則混成單聲道）、有 `trim_db=` 就修掉頭尾靜音、正規化到 `peak=` 或 `lufs=`
（不能兩個都給）、以 `subtype=` 寫出（預設 `PCM_16`），在 `dst` 底下照原本的樹狀結構
放。清單則以檔案共同的最深資料夾為準，或以 `root=` 為準。輸出檔名已被佔用時會換一個，
所以要重做就先清空 `dst`，或傳 `overwrite=True`。`lufs=` 之下短於 0.4 s 的檔案會失敗，
只有一個大聲暫態的小聲檔案可能被推過滿刻度。

```python
result = kudio.convert_folder("incoming/", "clean/")
print(result, bool(result), [path.name for path, reason in result.failed])
```

```text
2/3 written, 1 failed False ['broken.wav']
```

失敗的檔案以 `(path, reason)` 放進 `failed`，其他檔案照做；這時結果為假，
`on_error="raise"` 則在第一個失敗就停下。資料夾工具只看 `.wav`。`kudio convert` 有檔案
失敗時結束碼是 1。

## 挑一個沒被佔用的名稱 {#picking-a-name-that-is-free}

```python
from pathlib import Path

for name in ["new.wav", "rec.wav", "take.v2.wav", "take.flac", "take.v2"]:
    print(name, "->", Path(kudio.check_file(Path("takes") / name)).name)
print(Path(kudio.check_file("takes/take.v2.wav", sep="_")).name)
```

```text
new.wav -> new.wav
rec.wav -> rec2.wav
take.v2.wav -> take.v22.wav
take.flac -> take2.wav
take.v2 -> take.v22.wav
take.v2_2.wav
```

[`check_file`][kudio.core.io.check_file] 一律傳回 `.wav` 檔名：`.flac` 這類音訊副檔名會被
換掉，點後面的其他東西算是名稱的一部分，計數從資料夾裡最大的號碼往上加。它會建立上層
資料夾，但不建立檔案，所以要在 `save_wave` 之前才問。[`check_path`][kudio.core.io.check_path]
對資料夾做同樣的事。

## 由 GUI 管理的片段 {#clips-kept-by-a-gui}

[`AudioDataManager`][kudio.core.manager.AudioDataManager] 是 GUI 片段清單背後的登記表。
`save_all(dst_path=...)` 寫出每個片段；兩個片段會寫到同一個檔案時，什麼都不寫並傳回
`False`。
