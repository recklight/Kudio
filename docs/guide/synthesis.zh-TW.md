# 建立含雜訊資料集 {#building-a-noisy-dataset}

[`Synthesizer`][kudio.core.synth.Synthesizer] 用乾淨語音和雜訊做出訓練用的配對，並依訊號遇到
的順序，加上你要的那段從說話者到檔案的路徑：

```text
talker -> room (rt60=) -> + noise (snr_ratio=) -> link (channel=) -> file
```

隨機的部分（雜訊從哪裡裁、房間、掉的封包）每個輸出檔案各有自己的種子，
[`manifest()`][kudio.core.synth.Synthesizer.manifest] 記下每個混音來自哪個乾淨檔案。記憶體裡
只有一段片段時，[`add_noise_snr`][kudio.effects.augment.add_noise_snr] 就夠了。

??? example "本頁範例使用的語料"

    兩個說話者的三段短語句、一段嘶聲和一段市電嗡聲。

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

[`syn()`][kudio.core.synth.Synthesizer.syn] 每個檔案傳回一個 `(output, clean, noise, snr)`；
房間、鏈路和目標訊號記在 [manifest](#the-manifest)。檔名是 `{clean}_{noise}_{snr}.wav`：
+5 寫成 `5dB`，−5 寫成 `n5`，2.5 寫成 `2p5dB`，有房間時再加上 `_rt300ms`。

!!! warning "`out_path` 會先被清空"
    預設 `overwrite=True` 時，`syn()` 寫檔前會刪掉 `out_path` 裡的所有東西。資料夾裡有輸入檔、
    是工作目錄或家目錄，或是連結或檔案系統根目錄時，它改丟
    [`SynthesisError`][kudio.exceptions.SynthesisError]。刪除之前，它會先在不碰樣本的情況下
    把整個執行試過一遍。`overwrite=False` 時不動非空的資料夾，發出警告並傳回 `None`。

| `mode=` | 寫出的檔案 | 用途 |
|---|---|---|
| `"regular"` / `"reg"`（預設） | 每個乾淨檔案一個，條件輪流分配 | 和乾淨語料一樣大的訓練集 |
| `"increment"` / `"inc"` | 乾淨 × 雜訊 × SNR | 依條件拆開分析的測試集 |

兩種模式都會再乘上房間數。`rdn_choice_num=N` 隨機挑 N 個乾淨檔案來合成。

## 資料夾、清單與 `root=` {#folders-lists-and-root}

`clean` 和 `noise` 收 [`check_input`][kudio.core.io.check_input] 收的東西：資料夾、`.wav`
檔、`.txt` 清單，或由這些組成的清單，所以像 `sorted(Path("data/clean").rglob("sx*.wav"))`
這樣的子集合不必先複製出來就能合成。輸出會在乾淨檔案的資料夾、清單共同的最深資料夾，或
`root=` 底下照原本的結構放，所以兩個說話者的 `sx133` 不會撞名。雜訊只以檔名主幹辨識，所以來自
不同資料夾的雜訊檔要先改成不同的名字。`syn(mkdir_parents=False)` 全部平鋪寫出。

## SNR 是相對於什麼量的 {#what-the-snr-is-measured-against}

```python
syx = kudio.Synthesizer("data/clean/f01/sa1.wav", "data/noise/hiss.wav",
                        out_path="data/check", snr_ratio=5, subtype="FLOAT")
[(out, clean, noise, snr)] = syx.syn(seed=17)
speech, sr = kudio.file_load(clean)
mix, sr = kudio.file_load(out)
talking = speech != 0                     # the pauses are digital zero
print(f"whole file {kudio.snr(speech, mix):.2f} dB, "
      f"while talking {kudio.snr(speech[talking], mix[talking]):.2f} dB")
```

```text
whole file 5.00 dB, while talking 6.54 dB
```

雜訊去掉平均值後，依乾淨檔案整段（含停頓）的平均功率縮放，所以有 30% 時間沒說話的人，說話
時會比標示的 SNR 高 1.5 dB；如果標示應該描述語音本身，請先修剪乾淨檔案。SNR 介於 -300 到
3000 dB。較短的雜訊會循環，較長的從依種子決定的位置裁切，並重新取樣到乾淨檔案的取樣率。
沒有任何變化的雜訊或乾淨檔案會丟出 `SynthesisError`。

## 房間與鏈路 {#rooms-and-links}

```python
from pathlib import Path

syx = kudio.Synthesizer("data/clean", "data/noise", out_path="data/rooms",
                        snr_ratio=(0, 5), rt60=(0.3, 0.8), drr_db=3.0,
                        write_targets=True)
syx.syn(mode="reg", seed=17)
kudio.save_manifest("data/rooms.json", syx.manifest())
for p in syx.manifest()[:3]:
    dry, wet, mix = (kudio.file_load(f)[0] for f in (p.clean, p.target, p.noisy))
    noise = mix - wet
    print(f"{Path(p.noisy).name:26} against wet {kudio.snr(wet, wet + noise):5.2f} dB,"
          f" against dry {kudio.snr(dry, dry + noise):5.2f} dB")
```

```text
sa1_hiss_0dB_rt300ms.wav   against wet  0.00 dB, against dry  1.92 dB
sa1_hiss_0dB_rt800ms.wav   against wet  0.00 dB, against dry  0.94 dB
sx133_hum_5dB_rt300ms.wav  against wet  5.00 dB, against dry  5.35 dB
```

`rt60=` 收一個或多個殘響時間，在加雜訊之前讓每個乾淨檔案經過它自己的房間
（[`rir`][kudio.core.room.rir]、[`apply_rir`][kudio.core.room.apply_rir]），距離由 `drr_db=`
（0 dB）決定。雜訊要和房間留下的聲音競爭，所以 SNR 是相對於有殘響的語音設定的。manifest 的
目標是乾淨檔案；`write_targets=True` 會另外把有殘響的乾淨訊號寫到 `out_path/TARGETS/`，給不該
負責消除房間效應的模型用。

`channel=` 讓每個完成的混音經過一條鏈路，可以是 [`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS]
的預設或一個 dict（[房間與通道](degradation.md)），`TARGETS/` 不受影響。它是整次執行的一個
設定，不是一個軸，manifest 的每一列都會標上它。

## 種子與多行程 {#seeds-and-the-pool}

`seed=` 是基底：每個檔案的裁切位置、房間和掉的封包，都由基底加上它在 `out_path` 底下的路徑
決定，所以同樣的要求會逐位元組重建同樣的檔案，不論寫出的順序或由哪個行程寫。沒給種子時，基底
從 numpy 的全域亂數產生器抽，並保存在 [`.seed`][kudio.core.synth.Synthesizer.seed]：

```python
np.random.seed(2026)                       # only so the number below repeats
first = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run1",
                          rt60=0.5, channel="voip")
first.syn(mode="inc")                      # no seed
again = kudio.Synthesizer("data/clean", "data/noise", out_path="data/run2",
                          rt60=0.5, channel="voip")
again.syn(mode="inc", seed=first.seed)
run1, run2 = (sorted(Path(d).rglob("*.wav")) for d in ("data/run1", "data/run2"))
print("drew", first.seed, len(run1), "files, identical:",
      [f.read_bytes() for f in run1] == [f.read_bytes() for f in run2])
```

```text
drew 942082305 18 files, identical: True
```

`is_pool=True` 在一次執行超過 200 個檔案時，用 8 個工作行程寫出，位元組和單一行程的結果
相同。在 Windows 和 macOS 上，這個呼叫要放在 `if __name__ == "__main__":` 底下。

## 寫出檔案 {#writing-the-files}

```python
y, sr = kudio.file_load("data/clean/f01/sa1.wav")
kudio.save_wave("data/loud/sa1.wav", kudio.normalize(y, peak=0.9), sr)
for subtype in ("PCM_16", "FLOAT"):
    syx = kudio.Synthesizer("data/loud", "data/noise", out_path=f"data/loud_{subtype}",
                            snr_ratio=(0, 10), subtype=subtype)
    syx.syn(mode="inc", seed=17)
    print(subtype, sorted(round(peak, 2) for peak in syx.clipped.values()))
```

```text
PCM_16 [1.05, 1.26, 1.37, 2.07]
FLOAT []
```

`subtype=` 預設 `"PCM_16"`。超過滿刻度的混音在定點格式裡會被削峰，`syn()` 會發出警告，寫明
個數和最大峰值；[`.clipped`][kudio.core.synth.Synthesizer.clipped] 列出每個這樣的檔案和它的
峰值。`"FLOAT"` 保留原樣。`syn()` 的 `target_sr=` 會把輸出重新取樣；立體聲的乾淨檔案會混成
單聲道。

## manifest {#the-manifest}

[`save_manifest`][kudio.core.dataset.save_manifest] 把每個檔案寫成一個
[`Pair`][kudio.core.dataset.Pair]，存成純 JSON：`noisy`、`clean`、`noise`、`snr_db`、`rt60`、
`channel` 和 `target`；[`load_manifest`][kudio.core.dataset.load_manifest] 讀回來。只有 `noisy`
和 `clean` 是必要的，不認得的鍵會丟掉並發出警告。路徑用正斜線寫，輸入是相對路徑時就是相對
路徑，以執行腳本的資料夾為準。`is_silence=True` 會再把大約 `p_silence`（2%）的輸出寫成只有
雜訊的 `..._n00.wav`，`syn()` 會傳回它們，manifest 則不收。

## 訓練、驗證與測試 {#train-validation-and-test}

```python
pairs = kudio.load_manifest("data/mixed.json")
train, val, test = kudio.split_pairs(pairs, 0.2, 0.2, seed=17)
shared = {p.clean for p in train} & {p.clean for p in test}
print(len(train), len(val), len(test), len(shared), "clean files in train and test")
```

```text
10 4 4 2 clean files in train and test
```

[`split_pairs`][kudio.core.dataset.split_pairs] 以 `seed=` 洗牌（預設 `None`，每次切法都不同），
分的是列，不是說話者或雜訊，所以同一個乾淨檔案的三個 SNR 可能分別落在訓練集和測試集。要測
模型沒聽過的說話者或雜訊，請先切分乾淨檔案或雜訊檔，再用 `root=` 把每一份合成到各自的資料夾。
完全重複的列會丟掉並發出警告；指向同一個混音、內容卻不一致的兩列會丟出 `ValueError`。

## 背景上的雜訊，以及命令列 {#noise-over-a-background-and-the-command-line}

[`syn_extra_mode(background_path, background_snr)`][kudio.core.synth.Synthesizer.syn_extra_mode]
分兩階段混音：先把每個雜訊和每個背景以每個 `background_snr` 混合，再把每個乾淨檔案和這些混合、
每個雜訊、每個背景以每個 SNR 混合。它不套用房間和鏈路，平鋪寫出，−5 寫成 `n5dB`，也不收
`target_sr=`。`kudio synth --clean data/clean --noise data/noise --out data/cli --snr -5 0 5
--seed 17` 在 shell 裡執行一次合成，預設 `--mode inc`，並印出用的種子。[降噪](enhancement.md)和
[品質評估](metrics.md)接著拿混音和它們的乾淨參考訊號往下做。
