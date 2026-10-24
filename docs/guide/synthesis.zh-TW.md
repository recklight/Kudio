# 建立含雜訊資料集 {#building-a-noisy-dataset}

降噪模型是從成對的資料學的：一段劣化的錄音，以及它背後的乾淨訊號。
[`Synthesizer`][kudio.core.synth.Synthesizer] 用乾淨語音和雜訊做出這樣的配對，
並照你的要求，把說話者到檔案之間的路徑加進去，要加多少就加多少，
順序和訊號實際經過的一樣：

```text
talker -> room (rt60=) -> + noise (snr_ratio=) -> link (channel=) -> file
```

房間作用在說話者身上，雜訊和房間留下的聲音一起到達麥克風，
鏈結再把麥克風收到的一切傳出去，雜訊也包括在內。這條鏈上隨機的部分
（雜訊從哪裡截取、房間、遺失的封包）每個輸出檔各有各的種子，
[`manifest()`][kudio.core.synth.Synthesizer.manifest] 則記下哪份混音來自哪個
乾淨檔案，之後就沒有人需要再從檔名去推敲。如果只是記憶體裡的一段片段，用
[`add_noise_snr`][kudio.effects.augment.add_noise_snr] 就夠了
（[編輯與效果](editing.md)）；`Synthesizer` 是給硬碟上的語料庫用的。

??? example "這一頁範例使用的語料庫"

    兩位說話者的三段簡短語句，加上一段嘶聲和一段市電嗡嗡聲。全部都是合成的，
    所以這一頁的每個數字，都能用這個區塊重建出來。

    ```python
    import numpy as np
    import kudio

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * sr)) / sr


    def talker(f0):
        """Nine harmonics of f0, sounding for 350 ms of every 500 ms."""
        voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
        return (0.05 * voiced * (t % 0.5 < 0.35)).astype(np.float32)


    kudio.save_wave("data/clean/f01/sa1.wav", talker(210), sr)
    kudio.save_wave("data/clean/f01/sx133.wav", talker(190), sr)
    kudio.save_wave("data/clean/m02/sx133.wav", talker(115), sr)

    kudio.save_wave("data/noise/hiss.wav",
                    (0.1 * rng.standard_normal(3 * sr)).astype(np.float32), sr)
    n = np.arange(2 * sr) / sr
    hum = sum(np.sin(2 * np.pi * 50 * k * n) / k for k in (1, 3, 5))
    kudio.save_wave("data/noise/hum.wav", (0.05 * hum).astype(np.float32), sr)
    ```

## 第一次執行 {#a-first-run}

```python
import kudio

syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/mixed",
                        snr_ratio=(-5, 0, 5))
rows = syx.syn(mode="inc", seed=17)
kudio.save_manifest("data/mixed.json", syx.manifest())
print(len(rows), "files")
for out, clean, noise, snr in rows[:4]:
    print(out.as_posix(), "<-", clean.name, noise.name, snr)
```

```text
18 files
data/mixed/f01/sa1_hiss_n5.wav <- sa1.wav hiss.wav -5
data/mixed/f01/sa1_hiss_0dB.wav <- sa1.wav hiss.wav 0
data/mixed/f01/sa1_hiss_5dB.wav <- sa1.wav hiss.wav 5
data/mixed/f01/sa1_hum_n5.wav <- sa1.wav hum.wav -5
```

[`syn()`][kudio.core.synth.Synthesizer.syn] 每寫一個檔案，就回傳一個
`(output, clean, noise, snr)` tuple，不管你另外開了什麼選項，都維持四個元素：
回傳的形狀如果會隨某個關鍵字參數改變，每個把它拆開來用的呼叫端都會壞掉，
所以房間、鏈結和目標改放在 [manifest](#the-manifest) 裡。檔名的格式是
`{clean}_{noise}_{snr}.wav`：+5 寫成 `5dB`，−5 寫成 `n5`，不加 `dB`。
小數用 `p` 表示（`2p5dB`、`n2p5`），因為檔名裡的小數點會被當成副檔名的開頭；
有房間時再加上 `_rt300ms`。

!!! warning "`out_path` 會先被清空"
    在預設的 `overwrite=True` 下，`syn()` 寫檔之前會刪掉 `out_path` 裡的所有東西。
    如果那個資料夾裡有某個輸入檔、工作目錄或家目錄，或它本身就是檔案系統的根目錄，
    它會以 [`SynthesisError`][kudio.exceptions.SynthesisError] 拒絕，這樣
    `out_path=` 多往上指了一層時就攔得下來。除此之外，裡面的東西都會消失。
    `overwrite=False` 則不動非空的資料夾，只發出警告並回傳 `None`，
    所以後面還要繼續跑的腳本，請檢查回傳值。

## 用哪種模式：`reg` 還是 `inc` {#which-mode-reg-or-inc}

| `mode=` | 寫出的檔案 | 什麼時候用 |
|---|---|---|
| `"regular"` / `"reg"`（預設） | 每個乾淨檔案一個 | 訓練集和乾淨語料庫一樣大，各種條件分散在其中 |
| `"increment"` / `"inc"` | 乾淨 × 雜訊 × SNR | 每段語句都要在每種條件下出現，例如之後要依 SNR 拆開來看的測試集 |

給了 `rt60=` 時，兩種模式的檔案數都會再乘上房間數。`reg` 輪流分配雜訊和 SNR，
所以不管兩者各有幾個，只要連續取「雜訊數 × SNR 數」那麼多個檔案，
每種組合都剛好出現一次。語料庫比這個數目小時，就會有組合沒出現：
這裡的三個乾淨檔案對上 2 種雜訊和 3 個 SNR，拿到的是 −5 和 5 dB 的嘶聲，
以及 0 dB 的嗡嗡聲。

`rdn_choice_num=N` 不會合成全部的乾淨檔案，而是隨機抽 N 個（不重複抽取）來合成；
N 比語料庫還大時，會全部使用，並發出警告。

## 資料夾、清單和 `root=` {#folders-lists-and-root}

`clean` 和 `noise` 接受 [`check_input`][kudio.core.io.check_input] 接受的任何東西：
資料夾、`.wav` 檔、列出路徑的 `.txt` 清單，或是由這些組成的清單
（規則在[音訊檔案](audio-files.md)）。只讀 WAV 檔，`.wav` 或 `.WAV` 都可以。
搜尋資料夾時，其他格式一律默默略過；清單裡不是現有 `.wav` 檔的項目，
則會略過並發出 `UserWarning`。想只合成一部分、又不想先把它們複製出來，
就用清單。

輸出會照乾淨檔案的資料夾結構排列。資料夾以它自己為起點；
清單或 `.txt` 檔，則以其中檔案共同所在、最深的那一層資料夾為起點，
所以兩位說話者都唸了 `sx133`，還是會得到兩個檔案：

```python
from pathlib import Path
import kudio

# every talker's sx files, and nothing else
subset = sorted(Path("data/clean").rglob("sx*.wav"))
syx = kudio.Synthesizer(subset, "data/noise", out_path="data/mixed_sx",
                        snr_ratio=0)
for out, *_ in syx.syn(seed=17):
    print(out.as_posix())
```

```text
data/mixed_sx/f01/sx133_hiss_0dB.wav
data/mixed_sx/m02/sx133_hum_0dB.wav
```

如果那個共同的資料夾比你要的更深，就傳 `root=`。同一位說話者的檔案，
共同所在的就是那位說話者自己的資料夾，所以單獨拿來合成時，
會直接平鋪在 `out_path` 裡；加上 `root="data/clean"` 之後，
就會放到 `out_path/f01/`。`root=` 沒有包含所有乾淨檔案時，
會以 `ValueError` 拒絕。

只有乾淨檔案這一邊會照結構排列。雜訊檔不論是在輸出檔名，
還是在 manifest 的 `noise` 欄位裡，都只用主檔名辨識，
所以不同資料夾裡有同名檔案的雜訊集，要先改成不同的名字
（以 DEMAND 為例，每個環境的資料夾裡都有 `ch01.wav` 到 `ch16.wav`，
請複製出來，改名成 `kitchen_ch01.wav` 之類）。否則 `syn()` 會警告有輸出檔同名，
最後寫入的雜訊會留下來，manifest 則每個都留一列，
卻沒有任何欄位能說明檔案裡實際是哪個雜訊。

[`syn(mkdir_parents=False)`][kudio.core.synth.Synthesizer.syn] 會把所有東西平鋪
寫進 `out_path`。這時如果兩個同名的乾淨檔案要求同一個輸出檔名，
先要的那個會拿到，另一個則從這次執行和 manifest 中剔除，
並有一則警告指出它。

## SNR 以什麼為基準 {#what-the-snr-is-measured-against}

雜訊會先去掉平均值，再縮放到功率等於乾淨訊號的平均功率除以 $10^{\mathrm{SNR}/10}$。
這個平均是對整個乾淨檔案取的，停頓也算在內。寫成 `FLOAT` 時，
混音減掉乾淨檔案，會逐樣本得回縮放後的雜訊，所以可以用
[`snr`][kudio.core.evaluator.snr] 直接拿硬碟上的檔案驗證：

```python
import kudio

syx = kudio.Synthesizer("data/clean/f01/sa1.wav", "data/noise/hiss.wav",
                        out_path="data/check", snr_ratio=5, subtype="FLOAT")
[(out, clean, noise, snr)] = syx.syn(seed=17)

speech, sr = kudio.file_load(clean)
mix, sr = kudio.file_load(out)
talking = speech != 0                     # the pauses are digital zero
print(f"whole file     {kudio.snr(speech, mix):.2f} dB")
print(f"while talking  {kudio.snr(speech[talking], mix[talking]):.2f} dB")
```

```text
whole file     5.00 dB
while talking  6.54 dB
```

這位說話者有 30% 的時間沒出聲，所以在說話的時候，SNR 比標示的高 1.54 dB。
頭尾的靜音也有同樣的效果，所以如果標示要描述的是語音本身，
請先裁切乾淨檔案（[語音活動與音高](speech.md)可以找出語音在哪裡）。

雜訊檔比搭配的乾淨檔案短時，會重複接到一樣長；比較長時，
則從一個由該檔種子決定的隨機位置截取。混音前，雜訊會先重新取樣到乾淨檔案的取樣率。
完全沒有變化的雜訊檔，也就是數位靜音或固定準位，會引發
[`SynthesisError`][kudio.exceptions.SynthesisError] 並指出是哪個檔案，
因為沒有任何增益能把一個常數調到某個 SNR。數位靜音的乾淨檔案也會同樣被拒絕：
雜訊是依語音來縮放的，沒有語音，混出來的就會是一個全是零、
卻標著 SNR 的檔案。

## 房間：`rt60=` 與殘響目標 {#rooms-rt60-and-reverberant-targets}

`rt60=` 接受一個或多個以秒為單位的殘響時間，在加入雜訊之前，
先讓乾淨訊號經過一個房間；房間用 [`rir`][kudio.core.room.rir] 建立，用
[`apply_rir`][kudio.core.room.apply_rir] 套用。`drr_db=` 設定直達聲與殘響比，
預設 0 dB，大約是一兩公尺遠（刻度在[房間與通道](degradation.md)）。
每個檔案都有自己的房間，從它的種子抽出來，因為整個語料庫重複用同一個房間，
模型就會把那個房間學起來。混音保留乾淨檔案的長度，所以兩者逐樣本對齊。

在 manifest 裡，目標仍然是乾淨檔案，所以從混音學到乾淨檔案的模型，
除了雜訊之外，還得把房間也去掉。如果不想要求這麼多，`write_targets=True`
會另外把有殘響但乾淨的訊號寫到 `out_path/TARGETS/`，相對路徑和它的混音相同：
也就是房間作用之後的語音，沒有雜訊，也沒有鏈結。沒有 `rt60=` 時，
它會警告並且什麼都不寫，因為乾淨檔案本身就是那個訊號。
`TARGETS/` 位在 `out_path` 裡面，所以任何把 `out_path` 當成資料夾來讀的東西，
都會把這些目標當成混音；請改從 manifest 取得混音。

和雜訊競爭的是房間作用後留下的聲音，所以 SNR 是對**有殘響的**語音量的：

```python
from pathlib import Path
import kudio

syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/rooms",
                        snr_ratio=(0, 5), rt60=(0.3, 0.8), drr_db=3.0,
                        write_targets=True)
syx.syn(mode="reg", seed=17)
kudio.save_manifest("data/rooms.json", syx.manifest())

for p in syx.manifest():
    dry, wet, mix = (kudio.file_load(f)[0]
                     for f in (p.clean, p.target, p.noisy))
    noise = mix - wet
    name = Path(p.noisy).relative_to("data/rooms").as_posix()
    print(f"{name:30} against wet {kudio.snr(wet, wet + noise):5.2f} dB,"
          f" against dry {kudio.snr(dry, dry + noise):5.2f} dB")
```

```text
f01/sa1_hiss_0dB_rt300ms.wav   against wet  0.00 dB, against dry  1.92 dB
f01/sa1_hiss_0dB_rt800ms.wav   against wet  0.00 dB, against dry  0.94 dB
f01/sx133_hum_5dB_rt300ms.wav  against wet  5.00 dB, against dry  5.35 dB
f01/sx133_hum_5dB_rt800ms.wav  against wet  5.00 dB, against dry  8.08 dB
m02/sx133_hiss_5dB_rt300ms.wav against wet  5.00 dB, against dry  5.89 dB
m02/sx133_hiss_5dB_rt800ms.wav against wet  5.00 dB, against dry  5.66 dB
```

如果改用沒有殘響的乾檔來量，同樣這六個標示會差 0.35 到 3.08 dB，
而且差多少每個檔案都不一樣。

## 電話線路或編解碼器：`channel=` {#a-phone-line-or-codec-channel}

`channel=` 讓每份完成的混音經過一條鏈結。鏈結可以用
[`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] 裡的預設組合名稱指定，
也可以是以 [`CHANNEL_DEFAULTS`][kudio.effects.channel.CHANNEL_DEFAULTS] 的鍵
組成的 dict（`{"preset": "voip", "loss": 0.1}`）。它不會動 `TARGETS/` 裡的檔案：
要求模型重現一個編解碼器，就不是降噪了。

鏈結不是要掃過的軸。`snr_ratio` 和 `rt60` 會逐一掃過，檔案數也跟著倍增；
`channel` 則是整次執行只有一個設定，也不會出現在檔名裡，
因為一份語料庫要嘛是透過電話線路錄的，要嘛不是。要比較兩種鏈結，
就執行兩次，寫到兩個資料夾。manifest 會在每一列標上鏈結：
`"telephone"`，或是上面那個 dict 對應的 `"mu_law+loss10%x4-hold"`。

## 種子，以及行程池為什麼改變不了檔案 {#seeds-and-why-the-pool-cannot-change-a-file}

`seed=` 是一個基底值，不是整次執行共用的某個產生器的種子。
每個檔案的雜訊截取位置、房間和封包遺失，都由這個基底值加上它在
`out_path` 底下的路徑決定（用路徑而不是檔名，因為這裡兩位說話者都有
`sx133.wav`，不該共用同一個房間）。所以同樣的要求會逐位元組重建出同樣的檔案，
寫入的順序、由哪個行程寫，都改變不了任何一個檔案。沒有 `seed=` 時，
基底值從 numpy 的全域亂數產生器抽取（這個產生器從不會被重設種子），
並記在 [`.seed`][kudio.core.synth.Synthesizer.seed] 裡，
所以沒人給過種子的執行也能重建：

```python
from pathlib import Path
import numpy as np
import kudio

# seeded only so that the number printed below is the same every time
np.random.seed(2026)
first = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run1",
                          rt60=0.5, channel="voip")
first.syn(mode="inc")         # no seed
print("drew", first.seed)

again = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run2",
                          rt60=0.5, channel="voip")
again.syn(mode="inc", seed=first.seed)

run1 = sorted(Path("data/run1").rglob("*.wav"))
run2 = sorted(Path("data/run2").rglob("*.wav"))
print(len(run1), "files, identical:",
      [f.read_bytes() for f in run1] == [f.read_bytes() for f in run2])
```

```text
drew 942082305
18 files, identical: True
```

`is_pool=True` 在一次執行超過 200 個檔案時，用 8 個工作行程寫檔，
200 個以下則依序寫。在這份語料庫上以 `inc` 模式搭配 `snr_ratio=range(-10, 26)`，
用行程池和依序執行得到的 216 個檔案逐位元組相同。在 Windows 和 macOS 上，
Python 會把每個工作行程當成全新的直譯器啟動，工作行程一開始會重新匯入你的腳本，
所以這個呼叫必須放在 `if __name__ == "__main__":` 底下。少了這道防護，
每個工作行程一啟動就失敗，行程池不斷補上新的，整個執行就卡住了。

## 寫出檔案 {#writing-the-files}

`subtype=` 決定檔案的寫法，預設是 `"PCM_16"`；WAV 檔裝不下的 subtype，
在建立 Synthesizer 時就會被拒絕。混音是相加的結果，可能超過滿刻度。
定點的 subtype 會把它削在滿刻度（[`save_wave`][kudio.core.io.save_wave]
寫入之前就先削掉，所以不會溢位繞回），`PCM_16` 還會捨入到最接近的量化階。
接著 [`syn()`][kudio.core.synth.Synthesizer.syn] 會發出警告，附上檔案數和最大的峰值，
[`.clipped`][kudio.core.synth.Synthesizer.clipped] 則把每個這樣的檔案對應到它的峰值。
`"FLOAT"` 會保留樣本原本的值：

```python
import kudio

y, sr = kudio.file_load("data/clean/f01/sa1.wav")
kudio.save_wave("data/loud/sa1.wav", kudio.normalize(y, peak=0.9), sr)

for subtype in ("PCM_16", "FLOAT"):
    syx = kudio.Synthesizer("data/loud", "data/noise",
                            out_path=f"data/loud_{subtype}",
                            snr_ratio=(0, 10), subtype=subtype)
    syx.syn(mode="inc", seed=17)
    print(subtype, sorted(round(peak, 2) for peak in syx.clipped.values()))
```

```text
PCM_16 [1.05, 1.26, 1.37, 2.07]
FLOAT []
```

`PCM_16` 那次執行還會警告「4 of 4 file(s) went past full scale and were clipped
when written as PCM_16 (worst peak 2.07, +6.3 dBFS)」，
並建議降低乾淨檔案的準位，或改用 `subtype='FLOAT'`。

`syn()` 的 `target_sr=` 會重新取樣輸出；預設保留每個乾淨檔案自己的取樣率。
立體聲的乾淨檔案在讀入時就會混成單聲道。

## manifest {#the-manifest}

每個檔案一個 [`Pair`][kudio.core.dataset.Pair]：混音、它背後的乾淨檔案，
以及中間做了什麼。[`save_manifest`][kudio.core.dataset.save_manifest]
把它們寫成單純的 JSON list，任何工具不用 kudio 也讀得了，
[`load_manifest`][kudio.core.dataset.load_manifest] 則把它們讀回來。
上面寫出的 `data/rooms.json` 其中一列：

```json
{
  "noisy": "data/rooms/f01/sa1_hiss_0dB_rt300ms.wav",
  "clean": "data/clean/f01/sa1.wav",
  "noise": "hiss",
  "snr_db": 0,
  "rt60": 0.3,
  "channel": null,
  "target": "data/rooms/TARGETS/f01/sa1_hiss_0dB_rt300ms.wav"
}
```

只有 `noisy` 和 `clean` 是必要的，不認得的鍵會被丟掉，
所以其他工具寫的 JSON list，只要每一列都有這兩個欄位就能載入。

!!! note "相對路徑維持相對"
    路徑以正斜線寫入，Windows 和 Linux、macOS 一樣讀得懂，
    所以在一個系統上做的 manifest，在其他系統上也能載入。
    相對的 `out_path` 或輸入，產生的仍然是相對路徑，而且是相對於執行腳本時
    所在的資料夾，不是相對於 JSON 檔。請在那個資料夾裡載入 manifest；
    如果會從別的地方讀取，就傳入絕對路徑，否則
    [`Pair.exists()`][kudio.core.dataset.Pair.exists] 會是 `False`。

在 [`syn()`][kudio.core.synth.Synthesizer.syn] 加上 `is_silence=True`，
還會把大約 `p_silence` 比例的輸出再寫一次純雜訊的版本，
命名為 `..._n00.wav`：預設是 2%，捨入成整數個檔案，而且至少一個，
所以第一次執行的 18 個輸出會附帶一個。這些檔案會出現在 `syn()` 的回傳值裡，
`snr_db` 為 0，但不在 manifest 裡：它們沒有語音，
所以當初為它們抽的那個乾淨檔案，並不是它們的參考訊號。

## 訓練、驗證與測試 {#train-validation-and-test}

[`split_pairs`][kudio.core.dataset.split_pairs] 用
`numpy.random.default_rng(seed)` 洗牌一次，再把各列切成
`(train, validation, test)`。請傳 `seed=`：預設的 `None` 每次呼叫都會切出
不同的結果。如果照比例切完會沒有東西可以訓練，就改成放棄驗證集和測試集。

```python
import kudio

pairs = kudio.load_manifest("data/mixed.json")
train, val, test = kudio.split_pairs(pairs, 0.2, 0.2, seed=17)
print(len(train), len(val), len(test))
shared = {p.clean for p in train} & {p.clean for p in test}
print(len(shared), "of 3 clean files in both train and test")
```

```text
10 4 4
2 of 3 clean files in both train and test
```

問題出在第二行。`split_pairs` 分的是列，不是說話者、語句或雜訊，
一個乾淨檔案配上三個 SNR 就是三列，可能落在不同的切分裡。
每一列測試資料都是模型沒看過的混音，但裡面的說話者和雜訊，模型通常都聽過：
這裡兩種雜訊在訓練集和測試集裡都有，4 列測試資料中有 2 列，
同樣的雜訊和 SNR 組合在訓練集裡也有。這樣的切分問的是模型有沒有學到
訓練混音以外的東西，對它從沒聽過的說話者或雜訊則什麼也說明不了。
要回答後者，請先切分乾淨檔案（或雜訊檔），再把每個切分各自合成到自己的資料夾；
用清單當輸入再加上 `root=`，就能寫成一個迴圈：

```python
from pathlib import Path
import kudio

talkers = {"train": ["f01"], "test": ["m02"]}
for split, who in talkers.items():
    files = [f for t in who
             for f in sorted(Path("data/clean", t).glob("*.wav"))]
    syx = kudio.Synthesizer(files, "data/noise", out_path=f"data/{split}",
                            root="data/clean", snr_ratio=(-5, 0, 5))
    syx.syn(mode="inc", seed=17)
    kudio.save_manifest(f"data/{split}.json", syx.manifest())
    print(split, len(syx.manifest()), "rows")
```

```text
train 12 rows
test 6 rows
```

同一列出現兩次（`snr_ratio=` 裡有重複的值就會這樣），會先被拿去訓練，
再被當成沒參與訓練的資料來評分，所以 `split_pairs` 會丟掉完全相同的重複列，
並發出警告。兩列指向同一份混音、其他地方卻有出入時，會引發 `ValueError`。

## 在背景上再疊雜訊：`syn_extra_mode` {#noise-over-a-background-syn_extra_mode}

[`syn_extra_mode(background_path, background_snr)`][kudio.core.synth.Synthesizer.syn_extra_mode]
分兩階段混音。先把每種雜訊以每個 `background_snr` 和每個背景檔混合；
再把每個乾淨檔案以每個 SNR，分別和這些混合結果、每個單純的雜訊，
以及每個背景混合。2 個乾淨檔案、2 種雜訊、1 個背景搭配一個 `background_snr`，
再加上 2 個 SNR，就是 2 × (2 + 2 + 1) × 2 = 20 列。其中 2%、至少一個檔案，
會再寫一次純雜訊的版本，所以這裡 `syn_extra_mode` 回傳 21 列，
manifest 裡則是 20 列。每個乾淨檔案也會原封不動地複製一份，
命名為 `{stem}_n0.wav`，只有語音，正好和只有雜訊的 `_n00` 相對；
這些複本既不在回傳值裡，也不在 manifest 裡。

輸出資料夾同樣會先清空，防護和 [`syn()`][kudio.core.synth.Synthesizer.syn] 一樣，
但其他地方不同：

- 不套用房間和鏈結。
- 正負兩種 SNR 的檔名都帶 `dB`，也就是 `n5dB` 和 `5dB`，而 `syn()` 寫的是 `n5`。
  硬碟上現有的語料庫兩種寫法都有，所以兩邊都不改；
  請用 manifest 的 `snr_db` 比對，不要用檔名。
- 輸出是平鋪的：同名的乾淨檔案，例如兩位說話者的 `sx133.wav`，
  會寫到相同的輸出檔名。它會發出警告，後一個會覆寫前一個，
  所以給這個模式的乾淨檔案，名字請不要重複。
- 它不接受 `target_sr=`：每個輸出都維持乾淨檔案自己的取樣率，
  不管同一個 Synthesizer 之前的 `syn()` 要求過什麼。雜訊則和 `syn()` 一樣，
  以它要混入的那個檔案的取樣率讀進來。
- 輸出超過 200 個時，除非傳 `is_pool=False`，否則會用 8 個行程的行程池寫檔，
  這時在 Windows 和 macOS 上就需要 `if __name__ == "__main__":` 這道防護。

## 從命令列 {#from-the-command-line}

`kudio synth` 不必寫腳本就能建立同樣的語料庫。它的 `--mode` 預設是 `inc`，
而 [`syn()`][kudio.core.synth.Synthesizer.syn] 預設是 `regular`。
它會先清空 `--out`，防護和 `syn()` 一樣，並印出用到的種子，
所以沒有給 `--seed` 的執行，也能用它印出來的數字重建：

```bash
kudio synth --clean data/clean --noise data/noise --out data/cli \
    --snr -5 0 5 --seed 17 --rt60 0.3 0.8 --targets --manifest data/cli.json
```

這會寫出 36 份混音、`data/cli/TARGETS/` 底下的 36 個目標，以及 manifest。
所有選項都列在[命令列](../cli.md)那一頁。

接下來交給[降噪](enhancement.md)和[品質評估](metrics.md)：混音是輸入，
manifest 裡的乾淨檔案則是評分時對照的參考訊號。
