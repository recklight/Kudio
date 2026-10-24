# 命令列 {#command-line}

安裝 kudio 時也會一起裝好 `kudio` 指令，給那些不需要寫腳本的工作用：這段錄音能不能用、
有多大聲、語音在哪裡、經過電話線路會變成什麼樣子，以及轉換資料夾、建立含雜訊資料集這類
整批的工作。每個子指令都是呼叫函式庫，再把回傳的結果印出來；下面每一節都附上實際做事的
那個函式的連結，所有參數都在那裡說明。

```bash
kudio --help                        # every subcommand, one line each
kudio report --help                 # the options of one of them
python -m kudio.cli info take.wav   # when the kudio script is not on PATH
```

| | 指令 | 回答的問題 | 建立在 |
|---|---|---|---|
| 檢查錄音 | [`info`](#info) | 這是什麼檔案？ | [`audio_info`][kudio.core.io.audio_info] |
| | [`report`](#report) | 技術上有沒有問題？ | [`audio_report`][kudio.core.report.audio_report] |
| | [`loudness`](#loudness) | 有多大聲，音量在哪裡變了？ | [`loudness`][kudio.core.loudness.loudness]、[`loudness_over_time`][kudio.core.loudness.loudness_over_time]、[`true_peak`][kudio.core.loudness.true_peak] |
| | [`vad`](#vad) | 語音在哪裡？ | [`vad`][kudio.core.vad.vad] |
| | [`pitch`](#pitch) | 人聲在什麼音高，哪些地方有聲？ | [`f0`][kudio.core.pitch.f0] |
| 轉換與編輯 | [`convert`](#convert) | 重新取樣或重新編碼一個檔案或資料夾 | [`convert_folder`][kudio.core.io.convert_folder] |
| | [`normalize`](#normalize) | 把檔案調到指定的響度或峰值 | [`normalize_lufs`][kudio.core.loudness.normalize_lufs] |
| | [`trim`](#trim) | 切掉頭尾的靜音 | [`trim_silence`][kudio.effects.silence.trim_silence] |
| 房間與通道 | [`room`](#room) | 量測房間、建立房間，或把檔案放進房間 | [`rt60`][kudio.core.room.rt60]、[`rir`][kudio.core.room.rir] |
| | [`channel`](#channel) | 電話線路或有損傳輸會對檔案做什麼 | [`telephone`][kudio.effects.channel.telephone]、[`dropouts`][kudio.effects.channel.dropouts] |
| 建立含雜訊資料集 | [`synth`](#synth) | 從乾淨資料夾和雜訊資料夾做出含雜訊的語料庫 | [`Synthesizer`][kudio.core.synth.Synthesizer] |
| 降噪 | [`enhance`](#enhance) | 對一個檔案降噪 | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| | [`compare`](#compare) | 哪種降噪方法適合這個檔案？ | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| 對齊兩段錄音 | [`align`](#align) | 兩個版本差多遠，又因此損失多少？ | [`find_delay`][kudio.core.align.find_delay] |
| 音訊裝置 | [`devices`](#devices) | 有哪些音訊裝置？ | [`list_devices`][kudio.util.check.list_devices] |

## 所有指令的共同點 {#what-every-command-has-in-common}

- 音訊以檔案本身的取樣率讀入，而且大部分指令讀成單聲道：立體聲檔案在量測或寫出之前，
  會先平均成一個聲道，所以對立體聲檔案做 `convert` 會寫出單聲道檔。`info` 照實回報
  聲道數，`loudness` 和 `normalize` 則保留各聲道，因為 BS.1770 會把各聲道加權相加，
  立體聲檔案混成單聲道後量起來會低 3 LU。
- 單一輸入可以是 libsndfile 讀得了的任何格式，WAV 和 FLAC 都行。資料夾模式和 `synth`
  會蒐集每個子資料夾裡結尾為 `.wav` 的檔案，不分副檔名大小寫，所以錄音機產生的
  `ZOOM0001.WAV` 在任何系統上都找得到。
- 除非指令有 `--subtype`，輸出都是 16-bit PCM，定點格式的輸出超過滿刻度的部分會被削掉。
  單一輸出的格式由檔名決定，所以 `out.flac` 是 FLAC 檔，不能用 `--subtype FLOAT`；
  資料夾模式一律寫出 `.wav`。
- 單一輸出檔會直接被取代，不會先問。`convert --recursive` 從不取代任何東西：已經存在的
  輸出會改用新檔名（`mic_a2.wav`）。`synth` 剛好相反，會先清空它的輸出資料夾。
- 結果輸出到標準輸出，連同結果下方的摘要行，以及指令什麼都沒找到時印的那一行訊息
  （`no speech detected`）。錯誤與拒絕、記錄、警告和進度條則輸出到標準錯誤。所以
  `kudio vad take.wav > spans.txt` 會留下區段和它們的摘要，而腳本應該檢查結束代碼，
  不要去檢查檔案。
- `-v` 會把 kudio 的記錄調到 DEBUG，發生錯誤時顯示 Python 的 traceback 而不是一行訊息，
  而且要放在子指令前面：`kudio -v report take.wav`。放在後面的話，參數解析器會拒絕它。
  其他函式庫的記錄維持原樣。

## 範例檔案 {#the-example-files}

這一頁的每個輸出都是指令實際印出來的，跑在幾個合成檔案上。`take.wav` 是 16 kHz、
十秒長：四段音高在 105 到 170 Hz 之間滑動的有聲語句，第三段小 12 dB，中間有停頓和
環境音。`hot.wav` 是同一段錄音放大五倍並削峰，`excerpt.wav` 是從 4.2 秒開始的四秒，
`silence.wav` 是三秒的零。`noisy.wav` 以 5 dB SNR 加入白雜訊，`processed.wav` 是
`noisy.wav` 降噪後再延遲 128 個樣本。`raw/` 在兩個子資料夾裡放了三段 44.1 kHz 立體聲
24-bit 錄音，`corpus/` 則有三位說話者的乾淨錄音和兩種雜訊。

??? example "產生範例檔案"

    請在空資料夾裡執行。有幾個指令會讀前面指令寫出的檔案（`channel` 產生的 `phone.wav`、
    `synth` 產生的 `corpus/mixed`），所以請照頁面順序執行。

    ```python
    import numpy as np
    import kudio

    sr = 16000


    def phrase(seconds, f_start, f_end, level=1.0):
        """Voiced: 11 harmonics on a gliding pitch, four syllables a second."""
        t = np.arange(int(seconds * sr)) / sr
        phase = 2 * np.pi * np.cumsum(np.linspace(f_start, f_end, t.size)) / sr
        voice = sum(np.sin(k * phase) / k for k in range(1, 12))
        syllables = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)
        edges = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.03)    # no clicks
        return level * voice * syllables * edges / 6


    def take(seed=0):
        """Ten seconds: four phrases, the third 12 dB down, over room tone."""
        y = 0.003 * np.random.default_rng(seed).standard_normal(10 * sr)
        phrases = [(0.6, 1.8, 110, 140, 1.0), (3.0, 2.2, 150, 120, 1.0),
                   (6.0, 1.4, 115, 170, 0.25), (8.2, 1.3, 130, 105, 1.0)]
        for start, seconds, f_start, f_end, level in phrases:
            p = phrase(seconds, f_start, f_end, level)
            y[int(start * sr):int(start * sr) + p.size] += p
        return y.astype(np.float32)


    y = take()
    kudio.save_wave("take.wav", y, sr)
    kudio.save_wave("hot.wav", np.clip(5 * y, -1, 1), sr)       # recorded too hot
    kudio.save_wave("excerpt.wav", y[int(4.2 * sr):int(8.2 * sr)], sr)
    kudio.save_wave("silence.wav", np.zeros(3 * sr, np.float32), sr)

    hiss = np.random.default_rng(1).standard_normal(y.size).astype(np.float32)
    noisy = kudio.add_noise_snr(y, hiss, snr_db=5, seed=0)
    kudio.save_wave("noisy.wav", noisy, sr)
    denoised = kudio.spectral_enhance(noisy, sr)
    late = np.concatenate([np.zeros(128, np.float32), denoised])  # 8 ms late
    kudio.save_wave("processed.wav", late, sr)

    for i, name in enumerate(["day1/mic_a", "day1/mic_b", "day2/mic_a"]):
        mono = kudio.resample(take(seed=10 + i), sr, 44100)
        stereo = np.stack([mono, 0.8 * mono], axis=1)
        kudio.save_wave(f"raw/{name}.wav", stereo, 44100, subtype="PCM_24")

    for i in range(3):
        talker = 0.5 * take(seed=20 + i)
        kudio.save_wave(f"corpus/clean/spk{i + 1}.wav", talker, sr)
    t = np.arange(4 * sr) / sr
    fan = 0.1 * np.random.default_rng(5).standard_normal(t.size)
    hum = 0.1 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3))
    kudio.save_wave("corpus/noise/fan.wav", fan, sr)
    kudio.save_wave("corpus/noise/hum.wav", hum, sr)
    ```

## 檢查錄音 {#inspecting-a-recording}

這五個指令只讀檔案，不會改動它。能發現問題的那幾個，發現時會以 1 結束，這也是它們能用在
腳本裡的原因；完整清單在[最後一節](#exit-codes-and-shell-tests)。數字代表的意義，請看
[檢查錄音](guide/inspection.md)和[語音活動與音高](guide/speech.md)。

### info

檔頭的資訊，加上絕對值最大的樣本。這是唯一一個不動聲道的指令。

```bash
kudio info raw/day1/mic_a.wav
```

```text title="輸出"
file      : raw/day1/mic_a.wav
samplerate: 44100 Hz
channels  : 2
duration  : 10.000 s (441000 samples)
encoding  : WAV / PCM_24
peak      : 0.2922
```

### report

不需要乾淨參考訊號的健康檢查：準位、削峰、直流偏移、靜音、雜訊底、估計的 SNR，以及內容
佔滿的頻寬。它先印出 [`AudioReport`][kudio.core.report.AudioReport] 的一行摘要，再印出
[`problems()`][kudio.core.report.AudioReport.problems] 的發現，有任何發現就以 1 結束。

```bash
kudio report hot.wav
```

```text title="輸出"
10.00s @ 16000 Hz · peak 0.0 dBFS · -9.7 LUFS · SNR~33 dB · bandwidth 8000 Hz

· 3342 samples clipped (2.09% of the clip) — the recording was already too hot; no processing gets that back

(This is a technical check. A perceptual score with no reference needs a trained model — see kudio.dnsmos.)
```

對 `take.wav` 執行時，中間那行會是「Nothing wrong that a measurement can see.」，結束代碼
是 0。最後一行無論如何都會印，因為報告沒問題，不代表錄音聽起來沒問題；那要靠
[`dnsmos`][kudio.core.dnsmos.dnsmos]，它沒有自己的指令。

### loudness

依 BS.1770 的整合響度、依 EBU Tech 3342 的響度範圍、最大聲和最小聲的 3 秒視窗以及它們
的中心時間，還有真峰值，讀取前至少超取樣到 192 kHz。最小聲的視窗會略過數位靜音，
因為停頓不算發現。

```bash
kudio loudness take.wav
```

```text title="輸出"
integrated    -23.09 LUFS
range           6.68 LU
short-term    -22.82 LUFS at    3.40s  (loudest)
              -36.90 LUFS at    6.70s  (quietest)
true peak     -10.60 dBTP
```

| 選項 | 預設 | 作用 |
|---|---|---|
| `--over-time` | 關 | 印出曲線：每 100 ms 一行，-60 LUFS 以上每 2 LU 一個 `#` |
| `--momentary` | 關 | 極值和曲線改用 400 ms 視窗，而不是 3 秒 |
| `--target LUFS` | 無 | 整合響度和目標差超過 1 LU，或檔案是靜音時，以 1 結束 |

`--over-time` 是用來找出一段錄音在哪裡飄掉，不用打開編輯器。在 `excerpt.wav` 上，
視窗從第二段語句移到小聲的第三段時，準位降了 10 LU：

```bash
kudio loudness excerpt.wav --over-time
```

```text title="輸出"
integrated    -25.98 LUFS
range           5.98 LU
short-term    -26.47 LUFS at    1.50s  (loudest)
              -36.90 LUFS at    2.50s  (quietest)
true peak     -10.69 dBTP

     1.50s    -26.47  #################
     1.60s    -26.50  #################
     1.70s    -27.27  ################
     1.80s    -27.60  ################
     1.90s    -27.94  ################
     2.00s    -29.15  ###############
     2.10s    -29.19  ###############
     2.20s    -30.88  ##############
     2.30s    -31.63  ##############
     2.40s    -32.46  ##############
     2.50s    -36.90  ###########
```

每一行標的是視窗的中心時間，所以用 3 秒視窗時，第一行在 1.50 秒。短於 3 秒的片段仍然會
得到整合響度和真峰值，範圍那一行會說它填不滿視窗；加 `--momentary` 時還會得到 400 ms
的極值，範圍那一行會說需要 3 秒。短於 400 ms，也就是一個 BS.1770 區塊，連整合響度都沒有，
指令會以錯誤結束。數位靜音的檔案會印出 `-inf` 並說明，立體聲檔案則照立體聲量測。

`--target` 用來檢查交付的音量：

```bash
kudio loudness hot.wav --target -23
```

```text title="輸出"
integrated     -9.70 LUFS
range           6.63 LU
short-term     -9.22 LUFS at    3.20s  (loudest)
              -22.92 LUFS at    6.70s  (quietest)
true peak       0.34 dBTP

13.30 LU over the -23 LUFS target
```

它會以 1 結束。檔案裡沒有任何樣本超過滿刻度，真峰值卻在 0 dBTP 以上：削峰後的波形在
樣本之間衝過頭了。

### vad

[`vad`][kudio.core.vad.vad] 找到的語音區段，以秒為單位。門檻是從檔案本身的雜訊底往上算，
所以同一個值在安靜的檔案和有嘶聲的檔案上都適用。`--threshold-db`（預設 8）是音框必須
高出雜訊底多少：小聲的說話者被漏掉時調低，一陣陣的房間雜訊被當成語音時調高。
`--split-to DIR` 還會把每個區段各寫成一個檔案，命名為 `<stem>_001.wav` 依序往下。

```bash
kudio vad take.wav
```

```text title="輸出"
    1     0.550 →    2.460  (1.910s)
    2     2.950 →    5.270  (2.320s)
    3     5.980 →    7.460  (1.480s)
    4     8.150 →    9.560  (1.410s)
4 span(s), 7.12s of 10.00s (71%)
```

在 `noisy.wav` 上它找到三段，不是四段。第三段語句比其他的小 12 dB，在 5 dB SNR 下就落到
雜訊底下了。沒有語音的檔案會印出 `no speech detected` 並以 1 結束。

### pitch

[`f0`][kudio.core.pitch.f0] 曲線的摘要：中位數、以 Hz 和半音表示的第 5 到第 95 百分位數
範圍，以及檔案有多少比例是有聲的。用百分位數而不是極值，是因為一次八度誤判就能單獨決定
一個極值。

```bash
kudio pitch take.wav --segments --labels voiced.txt
```

```text title="輸出"
median 128.5 Hz  ·  111.2-158.9 Hz (6.2 semitones)  ·  71% voiced
    1     0.560 →    2.460  (1.900s)
    2     2.960 →    5.250  (2.290s)
    3     5.970 →    7.450  (1.480s)
    4     8.160 →    9.550  (1.390s)
wrote 4 label(s) to voiced.txt
```

`--labels` 用 [`save_labels`][kudio.core.dataset.save_labels] 寫出有聲區段。`voiced.txt`
是 Audacity 的標籤軌（開始、結束、文字，以 tab 分隔），所以在編輯器裡這些區段會疊在波形
上面；檔名是 `.txt` 或 `.tsv` 時用這個格式，其他副檔名則寫成 JSON。完全沒有有聲音框時，
指令會以 1 結束：對 3 秒的 `silence.wav` 它會印出 `no voiced frames in 301 (65-400 Hz)`，
301 是音框數。

`--fmin` 和 `--fmax`（65 和 400 Hz）設定搜尋範圍。`--hop` 的單位是樣本，不是毫秒；不給的話
是 10 ms，在 16 kHz 是 160 個樣本。`--frame-length` 必須容納超過兩個 `--fmin` 的週期，
否則指令會以錯誤結束。會有這個檢查，是因為不檢查的後果很隱蔽：週期放不進去的那些音高會
消失，剩下的看起來都正常，讀起來像是錄音本身的特性，而不是設定錯了。不給的話是 2048 個
樣本，在 88.2 和 96 kHz 則是 4096，因為那裡 65 Hz 的兩個週期需要 2714 和 2954 個樣本。

## 轉換與編輯 {#converting-and-editing}

用 Python 讀寫和轉換檔案，請看[音訊檔案](guide/audio-files.md)；裁切和音量則在
[編輯與效果](guide/editing.md)。

### convert

重新取樣和重新編碼。沒有 `--rate` 時檔案維持原本的取樣率，`--subtype` 接受任何 soundfile
支援的 subtype，預設是 `PCM_16`。

```bash
kudio convert raw/day1/mic_a.wav mic_a.wav --rate 16000
```

```text title="輸出"
wrote mic_a.wav (16000 Hz, PCM_16)
```

`mic_a.wav` 只有一個聲道：44.1 kHz 的立體聲原檔在讀入時就被平均了。加 `--recursive` 時，
兩個參數都是資料夾，工作交給 [`convert_folder`][kudio.core.io.convert_folder]，它會照來源
資料夾底下的結構建立輸出。`--lufs` 做響度正規化，`--trim-db` 裁掉靜音，兩者都在重新取樣
之後進行，單一檔案和資料夾都適用。

```bash
kudio convert raw/ 16k/ --rate 16000 --recursive --lufs -23
```

```text title="輸出"
3/3 file(s) -> 16k/
```

只要有任何檔案失敗，它就會以 1 結束，並列出前十個失敗的檔案和原因。再執行第二次時，
什麼都不會被取代：`16k/day1/mic_a.wav` 保留，新的輸出寫成 `16k/day1/mic_a2.wav`，
標準錯誤上的記錄會說明這件事。如果目的是重做一遍，請先清空資料夾。

### normalize

把檔案調整到指定的響度 `--lufs`（不給時是 -23），或指定的峰值 `--peak`，兩者不能同時用。
峰值是線性振幅，大於 0、最多 1，例如 0.99，不是 dBFS；其他的值會以錯誤結束，數位靜音的
檔案也一樣，因為它沒有準位可以縮放。各聲道會保留。要用耳朵比較兩個檔案之前，該用的是
響度，因為兩個版本中比較大聲的那個，不管其他地方有什麼差異，往往就會勝出。

```bash
kudio normalize hot.wav hot_r128.wav --lufs -23
kudio loudness hot_r128.wav --target -23 > /dev/null && echo "on target"
```

```text title="輸出"
wrote hot_r128.wav normalised to -23.0 LUFS
on target
```

正規化只是套用一個增益，不做別的。響度目標把峰值推過滿刻度時，
[`normalize_lufs`][kudio.core.loudness.normalize_lufs] 會在標準錯誤上說明，16-bit 檔案則會
被削峰；指令仍然以 0 結束。

```bash
kudio normalize take.wav loud.wav --lufs -6
```

```text title="stderr"
warning: normalize_lufs: -6.0 LUFS puts the peak at 2.09 (6.4 dBFS over); the signal will clip when written to an integer format
```

之後執行 `kudio report loud.wav` 會發現 12164 個削峰樣本，佔檔案的 7.60%。我讓增益維持
精確，而不是加上限制器，因為限制器會改變你要求的響度。

### trim

用 [`trim_silence`][kudio.effects.silence.trim_silence] 切掉開頭和結尾的靜音。
`--top-db`（預設 30）是從最大聲的音框往下算，不是從滿刻度。

```bash
kudio trim take.wav take_trim.wav --top-db 30
```

```text title="輸出"
trimmed [9216:153088] -> take_trim.wav (143872 samples)
```

方括號裡的數字是保留下來的樣本範圍。用 `--top-db 40` 時，同一段錄音會整段保留，
`[0:160000]`：它的環境音比最大聲的音框低 32 到 35 dB，落在 40 dB 以內，所以算內容。
像這樣的固定門檻適合安靜錄音室錄的音。對雜訊底明顯的檔案，請在 Python 裡用
[`vad_trim`][kudio.core.vad.vad_trim] 裁切，它依雜訊底設定門檻，保留第一段到最後一段
語音之間的所有內容。[`kudio vad`](#vad) 會顯示那一刀會落在哪裡，但沒有指令會實際寫出來。

## 房間與通道 {#rooms-and-channels}

兩種不是加成性雜訊的劣化。房間是摺積；電話線路或有損傳輸則會量化並遺失封包。
[房間與通道](guide/degradation.md)兩者都有說明。

### room

只給檔案、不加其他選項時，`room` 會把它當成脈衝響應，依 ISO 3382-1 量測。`--make` 則是
建立一個合成房間，並印出它所建立房間的量測結果，也就是下面的前兩行；量測那個檔案會再印出
同樣的摘要，以及背後的六個數值。`--apply` 把檔案放進房間裡。`--make` 同時又給檔案會被拒絕，
因為它不讀任何檔案。

```bash
kudio room --make room.wav --rt60 0.6 --drr 3 --seed 0
kudio room room.wav
```

```text title="輸出"
wrote room.wav (0.90s at 16000 Hz)
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T20            0.603 s
T30            0.598 s
EDT            0.583 s
C50            +9.24 dB
C80           +12.11 dB
DRR            +3.00 dB
```

[`rir`][kudio.core.room.rir] 會反推出能產生指定 DRR 的直達聲，所以用
[`rt60`][kudio.core.room.rt60] 量回這個房間，得到的就是建立它時用的數字。檔案長度是 RT60
的 1.5 倍。加 `--apply` 時，房間以輸入檔的取樣率建立，輸出則保留輸入的長度和峰值：

```bash
kudio room take.wav --apply take_room.wav --rt60 0.6 --drr 3 --seed 0
```

```text title="輸出"
wrote take_room.wav — 0.6s room, DRR +3 dB, at 16000 Hz
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
```

`--rt60`（預設 0.5 秒）和 `--drr`（預設 0 dB）描述要建立的房間。DRR +10 dB 以上是近距離
麥克風，0 dB 大約是一兩公尺遠。`--make` 以 `--rate`（預設 16000）建立，同一個 `--seed`
會建出同一個房間。

量測不是脈衝響應的東西，仍然會得到數字，因為反向積分會讓任何訊號都呈現衰減。指令在相信
這個衰減之前會先檢查，直線擬合的 R² 低於 0.9，或 EDT 和 T30 相差超過兩倍時，以 1 結束。
近距離的響應如果光是直達聲就掉了 10 dB，就沒有 EDT，這時只依擬合來判斷：

```bash
kudio room take.wav
```

```text title="輸出"
T30 41.05 s (EDT 60.73 s)  ·  C50 -15.5 dB  ·  DRR -24.5 dB  ·  fit 0.361  — but the decay is not straight, so read it with care
T20           48.208 s
T30           41.047 s
EDT           60.733 s
C50           -15.49 dB
C80           -14.74 dB
DRR           -24.50 dB
```

### channel

照傳輸鏈結的順序劣化一個檔案：先限頻和壓擴，再重新量化，最後是封包遺失。輸出的取樣率和
長度跟輸入一樣，摘要會比較兩者。

```bash
kudio channel take.wav phone.wav --telephone
```

```text title="輸出"
wrote phone.wav
  · telephone (mu_law)
bandwidth     8000 Hz ->    3797 Hz
si_sdr       -4.54 dB against the original
```

`--telephone` 就是 [`telephone`][kudio.effects.channel.telephone]：300 到 3400 Hz，降到
8 kHz，經過 G.711 mu-law 再升回來。加上 `--codec a_law` 則改用歐洲的壓擴曲線。之後
`kudio report phone.wav` 會以 1 結束，只有一項發現：「content stops at 3797 Hz although
the file claims 16000 Hz」，電話錄音就是這樣露出馬腳的。

封包遺失是 [`dropouts`][kudio.effects.channel.dropouts]。`--loss` 是要遺失的 20 ms 封包
比例，遺失的數量剛好就是這麼多，而不是照這個機率抽：500 個封包的 5% 是 25 個。獨立遺失時，
大部分封包是一次掉一個；`--burst N` 則改成平均 N 個封包一串，真實鏈結就是這樣掉封包的。

```bash
kudio channel take.wav lossy.wav --loss 0.05 --seed 0
kudio channel take.wav bursty.wav --loss 0.05 --burst 6 --seed 0
```

```text title="輸出"
wrote lossy.wav
  · 25 packet(s) lost in 23 run(s) of 1.1, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +14.03 dB against the original
wrote bursty.wav
  · 25 packet(s) lost in 6 run(s) of 4.2, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +23.84 dB against the original
```

這一行回報的是實際產生的串。這裡的長度是 4、6、1、1、4 和 9 個封包，六個隨機長度的平均
是 4.2 而不是 6。兩行 SI-SDR 之間將近 10 dB 的差距，來自封包落在哪裡，而不是成串遺失
本身造成的：六串只拿走錄音 0.4% 的能量，23 個單獨的遺失卻拿走 3.8%，換 30 個種子試，
成串的版本落在 9.8 到 26.3 dB 之間都有可能。同樣的道理，不要用 SI-SDR 那一行來挑
`--conceal` 模式。重複上一個封包（`hold`）量起來比留一個洞（`silence`）好還是壞，主要看
人聲的音高週期是否剛好能整除封包長度。

| 選項 | 預設 | 作用 |
|---|---|---|
| `--telephone` | 關 | 頻帶、取樣率和壓擴一起 |
| `--codec` | 搭配 `--telephone` 時為 `mu_law` | 用哪一條 G.711 曲線。搭配 `--telephone` 時選電話用的那條；單獨使用時只做壓擴，不限頻 |
| `--bits` | 無 | 線性重新量化成這麼多位元 |
| `--loss`、`--packet-ms` | 無、20 ms | 遺失封包的比例，以及封包長度 |
| `--burst` | 獨立 | 以封包數計的平均串長度，至少 1 |
| `--conceal` | `silence` | `silence` 留下一個洞，`hold` 重複上一個封包 |
| `--seed` | 無 | 同一個種子會遺失同樣的封包 |

這些可以疊加：`--codec a_law --bits 8 --loss 0.02` 三者都套用。一個都沒給時，指令什麼都
不寫，並以 2 結束。

## 建立含雜訊資料集 {#building-a-noisy-dataset}

### synth

透過 [`Synthesizer`][kudio.core.synth.Synthesizer]，把每個乾淨檔案以指定的 SNR 混入雜訊。
這個類別在[建立含雜訊資料集](guide/synthesis.md)有說明；這裡是同一件事的 shell 版本。

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/mixed \
    --snr -5 0 5 --seed 17
```

```text title="輸出"
synthesized 18 files -> corpus/mixed (seed 17)
```

預設的 `--mode inc` 會把每個乾淨檔案和每種雜訊以每個 SNR 各寫一份：三位說話者、兩種雜訊
和三個 SNR，共 18 個檔案，命名為 `spk1_fan_n5.wav`、`spk1_fan_0dB.wav`、
`spk1_fan_5dB.wav` 等等。`--mode reg` 則是每個乾淨檔案寫一份，輪流使用各種雜訊和 SNR。

| 選項 | 預設 | 作用 |
|---|---|---|
| `--clean`、`--noise` | 必填 | 一個資料夾、一個 `.wav`，或每行一個路徑的 `.txt` 檔 |
| `--out` | 必填 | 輸出資料夾，會先清空 |
| `--snr` | -5 0 5 | 以 dB 為單位的 SNR，可以有小數，例如 2.5 |
| `--seed` | 隨機抽取並印出 | 同一個種子會重建出同樣的檔案 |
| `--rt60` | 無 | 以秒為單位的殘響時間，每個值一個房間 |
| `--drr` | 0 dB | 這些房間的直達聲與殘響比，意義同 [`room`](#room) |
| `--targets` | 關 | 另外寫出沒有雜訊的殘響語音；需要 `--rt60` |
| `--channel` | 無 | 來自 [`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] 的鏈結，套用到每一份混音 |
| `--channel-loss` | 預設組合的值 | 取代那個鏈結的遺失比例；沒有 `--channel` 時會被拒絕 |
| `--channel-burst` | 預設組合的值 | 取代它的平均串長度；沒有 `--channel` 時會被拒絕 |
| `--manifest FILE` | 無 | 每個輸出檔一筆 JSON 紀錄 |

!!! warning "每次執行前都會清空 `--out`"
    輸出資料夾裡的東西會先被刪掉，這樣重跑時就不會有舊的檔案混在新的裡面。資料夾是磁碟
    根目錄，或裡面有某個輸入檔、工作目錄或家目錄時，指令會拒絕；除此之外，裡面的東西
    都會被刪掉。請給每份語料庫一個專屬的資料夾。

每個檔案的雜訊截取位置、房間和封包遺失，都是由種子加上它自己的路徑決定，所以同樣的指令
會逐位元組重建出同樣的檔案。沒有 `--seed` 時會隨機抽一個種子，印在第一行的最後；請把它
記下來，或傳入 `--seed` 並把它留在指令裡。

房間和鏈結照錄音實際遇到的順序套用：房間作用在說話者身上，雜訊在麥克風處加入，鏈結再
傳送整份混音。

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/rooms \
    --snr 0 5 --mode reg --rt60 0.3 0.6 --targets \
    --manifest corpus/rooms.json --seed 17
```

```text title="輸出"
synthesized 6 files -> corpus/rooms (seed 17)
rooms: 0.3s, 0.6s at DRR +0 dB — the SNR is against the reverberant speech, and the clean file is still the target
targets: 6 reverberant-clean file(s) -> corpus/rooms/TARGETS
manifest -> corpus/rooms.json
```

每個 `--rt60` 值都會讓輸出數量倍增，`reg` 和 `inc` 模式都一樣：三個乾淨檔案變成六個。
房間會寫進檔名（`spk1_fan_0dB_rt300ms.wav`）。`--targets` 把殘響語音以相同檔名寫到
`TARGETS/`，給只該去除雜訊、保留房間效果的模型用；沒有 `--rt60` 時沒有東西可寫，會有警告
說明。[`load_manifest`][kudio.core.dataset.load_manifest] 可以把 manifest 讀回來，每個檔案
一筆，包含路徑、雜訊、SNR、RT60、鏈結和目標。

`--channel` 接受 `telephone`、`telephone_a`、`mu_law`、`a_law`、`voip`（mu-law，5% 遺失、
平均 4 個一串，以重複上一個封包補償）或 `none`。`--channel voip --channel-loss 0.1
--channel-burst 6` 會改掉遺失設定、其餘保留，摘要中這個鏈結的名稱是
`mu_law+loss10%x6-hold`。整次執行只用一種鏈結，所以兩種鏈結就是兩份語料庫；原因在
[房間與通道](guide/degradation.md)。

檔案寫成 16-bit PCM，超過滿刻度的混音會被削峰，標準錯誤上會有一則警告統計檔案數。在 Python
裡，`Synthesizer` 還可以用 `subtype="FLOAT"`。

## 降噪 {#denoising}

每種增益規則和雜訊估計器的假設，以及如何在它們之間選擇，在[降噪](guide/enhancement.md)有
說明；`compare` 則是在你自己的檔案上做選擇的方法。

### enhance

用 [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] 對一個檔案降噪。預設是
MCRA 雜訊估計加上 log-MMSE，沒有理由選別的時候，我會從這組開始。

```bash
kudio enhance noisy.wav denoised.wav
kudio enhance noisy.wav denoised_omlsa.wav \
    --method omlsa --noise quantile --set q=0.4
```

```text title="輸出"
log-MMSE + mcra noise estimate -> denoised.wav
  noise floor -28.8 -> -48.1 dBFS
OM-LSA + quantile noise estimate -> denoised_omlsa.wav
  noise floor -28.8 -> -51.5 dBFS
```

`--method` 是 [`METHODS`][kudio.enhance.spectral.METHODS] 裡的七種增益規則之一（`specsub`、
`multiband`、`wiener`、`mmse_stsa`、`logmmse`、`omlsa`、`spectral_gate`），`--noise` 則是
[`NOISE_ESTIMATORS`][kudio.enhance.spectral.NOISE_ESTIMATORS] 裡的四種估計器之一（`mcra`、
`quantile`、`initial`、`minimum`）。`--n-fft`（預設 512）是音框長度，位移是它的四分之一。

`--set NAME=VALUE` 設定方法的參數，可以重複使用；方法沒有的參數名稱會以錯誤結束，並列出它
有的參數，超出 `METHODS` 所定範圍的值也一樣（`--set alpha=1.5`：「logmmse: alpha=1.5 is
outside 0.5 to 0.999」）。上面的 `q=0.4` 屬於 OM-LSA，是一個頻率格不含語音的先驗機率
（預設 0.30），不是 quantile 估計器的參數。估計器本身的設定，例如 `quantile` 取每個頻率格
功率的哪個分位數（0.15），無法從命令列設定，只能透過 `spectral_enhance` 的
`noise_kwargs`。

輸出長度和輸入相同，所以逐樣本對齊。第二行是處理前後的雜訊底；雜訊底比較低代表殘留雜訊
比較少，但不等於語音比較好。要知道降噪到底有沒有幫助，請用 [`compare`](#compare)。

加 `--recursive` 時兩個參數都是資料夾，
[`enhance_folder`][kudio.enhance.spectral.enhance_folder] 會把第一個資料夾底下的每個
`.wav` 降噪，照同樣的結構寫到第二個資料夾底下，使用相同的 `--method`、`--noise`、
`--n-fft` 和 `--set`：

```bash
kudio enhance corpus/mixed corpus/denoised --recursive
```

```text title="輸出"
18/18 file(s) -> corpus/denoised
  noise floor down 21.4 dB on average
```

只要有任何檔案失敗，它就會以 1 結束，並列出前十個失敗的檔案和原因。

### compare

對一個檔案跑過每一種方法，把結果放在同一張表。沒有參考訊號時，表上只有 `ms` 和雜訊底兩欄，
下方的說明會提醒這些是殘留雜訊的讀數，不是品質分數。以乾淨訊號作為 `--reference` 時，
表格依 SI-SDR 排序，`dSNR` 是相對於未處理輸入所增加的 SNR。`dSNR` 為負代表這個方法比
什麼都不做還糟，如果每一列都是負的，指令會直接說出來。

```bash
kudio compare noisy.wav --reference take.wav
```

```text title="輸出"
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
multiband + mcra                 29       -47.7     12.59    7.75      —   0.54
mmse_stsa + mcra                108       -43.7     12.22    7.41      —   0.53
logmmse + mcra                   60       -48.1     11.97    7.03      —   0.52
wiener + mcra                    39       -53.0     11.45    6.49      —   0.52
omlsa + mcra                     76       -50.7     11.22    6.26      —   0.52
specsub + mcra                   31       -39.0     11.15    6.47      —   0.53
spectral_gate + mcra             44       -41.4      6.14    0.28      —   0.54
```

輸入是合成的，所以請看這張表的樣子，不要當成這些方法在語音上的排名。雜訊底最低的
`wiener` 在這裡排第四。

雜訊估計器的影響和增益規則一樣大，所以 `--noises` 也會掃過它；`all` 代表全部四種。
`--noise` 則固定用一種，預設是 `mcra`，兩個同時給會被拒絕並以 2 結束。

```bash
kudio compare noisy.wav --reference take.wav \
    --methods logmmse wiener --noises all
```

```text title="輸出"
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
wiener + initial                 27       -45.7     15.50   10.62      —   0.53
logmmse + quantile               47       -48.6     14.95   10.01      —   0.53
logmmse + initial                54       -42.7     14.75    9.89      —   0.53
wiener + quantile                28       -53.2     14.75    9.79      —   0.52
logmmse + mcra                   61       -48.1     11.97    7.03      —   0.52
wiener + mcra                    38       -53.0     11.45    6.49      —   0.52
logmmse + minimum                86       -46.2      6.58    1.80      —   0.50
wiener + minimum                 64       -51.3      5.23    0.75      —   0.50
```

`initial` 從檔案最前面的音框取得雜訊，而 `noisy.wav` 開頭有 0.6 秒沒人說話的純雜訊，正是
它設計時針對的情況。把同一對檔案從 1.0 秒開始截，也就是落在第一段語句裡面，兩列 `initial`
就會掉到表格最底下，`dSNR` 是 -2.89 和 -3.44 dB，比什麼都不做還糟。

| 選項 | 預設 | 作用 |
|---|---|---|
| `--reference FILE` | 無 | 乾淨訊號，會重新取樣到輸入的取樣率。它必須對齊並涵蓋相同的範圍：先執行 [`align`](#align) |
| `--methods` | 全部七種 | 要執行的增益規則 |
| `--write-to DIR` | 無 | 把每個結果寫成 `<stem>_<method>.wav`，有多個估計器時寫成 `<stem>_<method>_<noise>.wav` |
| `--no-metrics` | 關 | 即使有安裝也略過 PESQ 和 STOI |

`compare` 檢查兩者長度的方式和 [`si_sdr`][kudio.core.evaluator.si_sdr] 一樣：長度差一點時，
以較短的那個評分並發出警告；3 秒的參考訊號對上 10 秒的 `noisy.wav`，有 70% 量不到，
會被拒絕。`align --write-to` 會寫出等長的一對。

`ms` 是每次執行的實際經過時間，在暖機之後才量，這樣第一個方法就不會被算進 librosa 一次性
的編譯時間。STOI 需要 `kudio[eval]`，PESQ 需要 `kudio[pesq]`（各組裝了什麼，請看
[入門](getting-started.md)）；沒有安裝時該欄顯示 `—`，標準錯誤上會有警告說明要裝什麼。
PESQ 欄用的是 kudio 預設的 MOS-LQO：16 kHz 是寬頻、8 kHz 是窄頻，其他取樣率則留空。
論文引用的 PESQ 各有不同，要把這一欄和論文並列之前請先確認；[品質評估](guide/metrics.md)
有兩種尺度的 [`pesq`][kudio.core.evaluator.pesq]。

## 對齊兩段錄音 {#lining-up-two-recordings}

### align

用 [`find_delay`][kudio.core.align.find_delay] 量一段錄音落後另一段多少。每個參考型指標都是
拿第 *i* 個樣本和第 *i* 個樣本比，兩個檔案錯開時，沒有一個指標能察覺，只會給出很差的分數。

```bash
kudio align take.wav processed.wav --metrics --write-to aligned/
```

```text title="輸出"
lags by 128 samples (8.0 ms), correlation 0.970
overlap     10.000s of 10.000s / 10.008s

si_sdr         -2.16 dB  as the files sit
               11.97 dB  aligned

wrote 2 file(s) to aligned
```

這裡八毫秒的偏移讓 SI-SDR 少了 14 dB。`--write-to` 會把兩者裁到共同的範圍，寫成
`take_aligned.wav` 和 `processed_aligned.wav`，長度相同。`lags` 表示第二個檔案比第一個晚
開始，`leads` 表示比較早，`already aligned` 表示最佳延遲為零；相關係數為負時會回報極性相反，
而不是幫你修正。兩個檔案取樣率不同時，第二個會先重新取樣成第一個的取樣率，再做任何量測。

互相關一定有最大值，所以任何兩個檔案都會算出一個延遲。要不要相信它，由相關係數決定：
低於 0.5（[`MIN_CORRELATION`][kudio.core.align.MIN_CORRELATION]）時，指令會說兩個檔案
無關，並以 1 結束。

```bash
kudio align take.wav corpus/noise/fan.wav
```

```text title="輸出"
leads by 11727 samples (732.9 ms), correlation 0.016
overlap     4.000s of 10.000s / 4.000s

these do not look like the same recording — the delay above does not mean anything
```

`--max-seconds` 限制兩個方向的搜尋範圍，避免又長又重複的檔案配對到半分鐘外的段落。請把它
設在你預期的最大延遲之上。如果設得比實際延遲還小，搜尋會回傳範圍內最好的延遲，而這個延遲
仍可能通過 0.5 的檢查：`kudio align take.wav processed.wav --max-seconds 0.005`（80 個
樣本，小於實際的 128）會回報延遲 2 個樣本、相關係數 0.621，並以 0 結束。

## 音訊裝置 {#audio-devices}

### devices

列出輸入和輸出裝置，需要 `kudio[audio]`，一律以 0 結束。兩套音訊後端的裝置編號各自獨立，
所以它會印出兩份清單，並標明哪些函式用哪一份的編號：sounddevice 的編號給
[`record`][kudio.core.stream.record]，PyAudio 的編號給串流類別。一份清單裡的編號在另一份
裡沒有意義。沒有安裝這組套件時，它會說缺了什麼：

```bash
kudio devices
```

```text title="輸出"
sounddevice indices — use with kudio.record(device=...)
  unavailable: 'sounddevice' is required for this feature. Install it with: pip install kudio[audio]

PyAudio indices — use with kudio.Recorder / stream readers
  unavailable: 'PyAudio' is required for this feature. Install it with: pip install kudio[audio]
```

裝了之後，每個裝置一行：編號、輸入和輸出的聲道數、預設取樣率和名稱。其餘的請看
[錄音與即時檢視](guide/live-audio.md)。

## 結束代碼與 shell 測試 {#exit-codes-and-shell-tests}

能發現問題的指令，發現時會以 1 結束：

| 指令 | 以 1 結束的時機 |
|---|---|
| `report` | 有任何發現：削峰、峰值在滿刻度 0.5 dB 以內或超過、峰值低於 -30 dBFS、直流偏移大於 0.01、內容在奈奎斯特頻率的 90% 以下就結束、靜音超過 60%、估計 SNR 低於 10 dB、數位靜音，或有 NaN 和 inf 樣本 |
| `loudness --target` | 整合響度和目標差超過 1 LU，或檔案是靜音 |
| `vad` | 找不到語音 |
| `pitch` | 沒有任何有聲音框 |
| `align` | 相關係數低於 0.5：不是同一段錄音 |
| `room FILE` | 衰減看起來不像脈衝響應 |
| `convert --recursive`、`enhance --recursive` | 有任何檔案失敗 |

結束代碼 2 代表錯誤，會在標準錯誤上以一行回報。包括用法錯誤（未知的選項、型別錯誤的值、
缺少參數、`enhance --set` 沒有 `=`、`compare` 同時給了 `--noise` 和 `--noises`、`channel`
沒有事可做）、讀不了的檔案，以及函式庫拒絕的參數。`-v` 會改為顯示 traceback。其餘情況
指令都以 0 結束，就算 `compare` 發現沒有方法勝過輸入，或 `normalize` 寫出了削峰的檔案也一樣。

所以每項檢查在 shell 腳本裡都只要一行。`phone.wav` 是 [`channel`](#channel) 那一節寫出的
檔案。

=== "bash"

    ```bash
    kudio report hot.wav > /dev/null || echo "needs another take"

    for f in take.wav hot.wav phone.wav silence.wav; do
        kudio report "$f" > /dev/null || echo "$f"
    done
    ```

=== "PowerShell"

    ```powershell
    kudio report hot.wav | Out-Null
    if ($LASTEXITCODE -ne 0) { "needs another take" }

    foreach ($f in "take.wav", "hot.wav", "phone.wav", "silence.wav") {
        kudio report $f | Out-Null
        if ($LASTEXITCODE -ne 0) { $f }
    }
    ```

```text title="輸出"
needs another take
hot.wav
phone.wav
silence.wav
```

三個檔案，三種不同的發現：削峰、電話線路的頻寬，以及裡面什麼都沒有。`||` 遇到結束代碼 2
也會觸發，所以 `kudio report missing.wav || echo "needs another take"` 對不存在的檔案也會
印出那則訊息；需要區分發現和錯誤的腳本，請分別檢查 1 和 2。
