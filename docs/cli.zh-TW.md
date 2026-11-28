# 命令列 {#command-line}

安裝 kudio 時也會裝上 `kudio` 指令，給不需要寫腳本的工作用。每個子指令呼叫函式庫並印出結果；
最後一欄的函式有每個參數的說明。

```bash
kudio --help                        # every subcommand, one line each
kudio report --help                 # the options of one of them
python -m kudio.cli info take.wav   # when the kudio script is not on PATH
```

| | 指令 | 回答什麼 | 建在 |
|---|---|---|---|
| 檢查錄音 | [`info`](#info) | 這是什麼檔案？ | [`audio_info`][kudio.core.io.audio_info] |
| | [`report`](#report) | 技術上有沒有毛病？ | [`audio_report`][kudio.core.report.audio_report] |
| | [`loudness`](#loudness) | 多大聲，音量在哪裡變了？ | [`loudness`][kudio.core.loudness.loudness]、[`loudness_over_time`][kudio.core.loudness.loudness_over_time]、[`true_peak`][kudio.core.loudness.true_peak] |
| | [`vad`](#vad) | 語音在哪裡？ | [`vad`][kudio.core.vad.vad] |
| | [`pitch`](#pitch) | 聲音的音高是多少？ | [`f0`][kudio.core.pitch.f0] |
| 轉換與編輯 | [`convert`](#convert) | 重新取樣或重新編碼一個檔案或資料夾 | [`convert_folder`][kudio.core.io.convert_folder] |
| | [`normalize`](#normalize) | 把檔案調到某個響度或峰值 | [`normalize_lufs`][kudio.core.loudness.normalize_lufs] |
| | [`trim`](#trim) | 修掉兩端的靜音 | [`trim_silence`][kudio.effects.silence.trim_silence] |
| 房間與通道 | [`room`](#room) | 量測房間、建立房間，或把檔案放進房間 | [`rt60`][kudio.core.room.rt60]、[`rir`][kudio.core.room.rir] |
| | [`channel`](#channel) | 電話線或有損鏈路對檔案做了什麼 | [`telephone`][kudio.effects.channel.telephone]、[`dropouts`][kudio.effects.channel.dropouts] |
| 建立含雜訊資料集 | [`synth`](#synth) | 從乾淨資料夾和雜訊資料夾做出含雜訊語料 | [`Synthesizer`][kudio.core.synth.Synthesizer] |
| 降噪 | [`enhance`](#enhance) | 對檔案或資料夾降噪 | [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] |
| | [`compare`](#compare) | 哪個降噪器適合這個檔案？ | [`compare_enhancers`][kudio.enhance.spectral.compare_enhancers] |
| 對齊兩段錄音 | [`align`](#align) | 兩個版本差多遠？ | [`find_delay`][kudio.core.align.find_delay] |
| 評整個測試集 | [`score`](#score) | 系統在整個測試集上表現如何？ | [`score_dataset`][kudio.core.dataset_score.score_dataset] |
| 音訊裝置 | [`devices`](#devices) | 有哪些音訊裝置？ | [`list_devices`][kudio.util.check.list_devices] |

## 每個指令都一樣的地方 {#what-every-command-has-in-common}

- 音訊以檔案本身的取樣率讀取，大部分指令讀成單聲道；`info` 照實回報聲道，`loudness` 和
  `normalize` 保留聲道。
- 單一輸入可以是 libsndfile 讀得了的任何格式，裝了 `compressed` 套件組還能讀
  [m4a、aac 和 webm](guide/audio-files.md#formats-soundfile-does-not-read)。資料夾模式和 `synth`
  收集每個子資料夾裡的 `.wav` 檔。
- 沒有 `--subtype` 的指令都寫 16 位元 PCM。單一輸出依檔名決定格式，已存在時直接取代；`convert
  --recursive` 會替已存在的輸出換名字；`synth` 會先清空輸出資料夾。
- 結果寫到標準輸出；錯誤、記錄、警告和進度條寫到標準錯誤。請檢查結束碼，不要檢查輸出。
- `-v` 放在子指令之前（`kudio -v report take.wav`）：DEBUG 記錄，錯誤時印出 traceback 而不是
  一行訊息。`kudio --version` 印出安裝的版本。

## 範例檔案 {#the-example-files}

這一頁的每個輸出，都是指令在這些合成檔案上實際印出的內容。`take.wav` 是 16 kHz、十秒，四句
有聲的語句，第三句小 12 dB；`hot.wav` 是它放大五倍再削峰，`excerpt.wav` 是其中四秒，
`silence.wav` 是三秒的零，`noisy.wav` 是它混上 5 dB SNR 的白雜訊，`processed.wav` 是那個再
降噪、晚 128 個樣本。`raw/` 有三個 44.1 kHz 立體聲錄音，`corpus/` 有三個說話者和兩種雜訊，
`testset/` 有 12 個說話者的 24 句，乾淨和有雜訊的各一份。

??? example "產生範例檔案"

    在空資料夾裡執行，再依頁面順序執行各個指令。

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

    for k in range(12):                    # testset/: 12 talkers, two sentences each
        for n in (1, 2):
            rng = np.random.default_rng(100 * k + n)
            low = 95 + 9 * k
            voice = np.concatenate([phrase(1.2, low, low + 20 * n), np.zeros(sr // 4),
                                    phrase(1.0, low + 25, low)])
            clean = 0.5 * voice + 0.003 * rng.standard_normal(voice.size)
            noisy = kudio.add_noise_snr(clean, rng.standard_normal(clean.size),
                                        snr_db=5, seed=0)
            kudio.save_wave(f"testset/clean/p{k + 1:02d}_{n:03d}.wav", clean, sr)
            kudio.save_wave(f"testset/noisy/p{k + 1:02d}_{n:03d}.wav", noisy, sr)
    ```

## 檢查錄音 {#inspecting-a-recording}

### info {#info}

```bash
kudio info raw/day1/mic_a.wav
```

```text
file      : raw/day1/mic_a.wav
samplerate: 44100 Hz
channels  : 2
duration  : 10.000 s (441000 samples)
encoding  : WAV / PCM_24
peak      : 0.2922
```

檔頭資訊，以及分塊讀出的峰值（[`audio_info`][kudio.core.io.audio_info]）。

### report {#report}

```bash
kudio report hot.wav
```

```text
10.00s @ 16000 Hz · peak 0.0 dBFS · -9.7 LUFS · SNR~33 dB · bandwidth 8000 Hz

· 3390 samples clipped (2.12% of the clip) — the recording was already too hot; no processing gets that back

(This is a technical check. A perceptual score with no reference needs a trained model — see kudio.dnsmos.)
```

[`audio_report`][kudio.core.report.audio_report] 的摘要和它的發現；有發現時結束碼是 1。檔案讀成
單聲道。

### loudness {#loudness}

```bash
kudio loudness take.wav
kudio loudness excerpt.wav --over-time
```

```text
integrated    -23.09 LUFS
range           6.68 LU
short-term    -22.82 LUFS at    3.40s  (loudest)
              -36.90 LUFS at    6.70s  (quietest)
true peak     -10.60 dBTP
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

整合響度、響度範圍和真峰值，以及最響和最靜的時刻；`--over-time` 加上短期響度曲線，每 100 ms
一行（`--momentary` 用 400 ms 視窗）。`--target -23` 在差超過 1 LU 時結束碼為 1。保留聲道。

### vad {#vad}

```bash
kudio vad take.wav
```

```text
    1     0.550 -    2.460  (1.910s)
    2     2.950 -    5.270  (2.320s)
    3     5.980 -    7.460  (1.480s)
    4     8.150 -    9.560  (1.410s)
4 span(s), 7.12s of 10.00s (71%)
```

以秒為單位的語音區段；`--split-to DIR` 把每段寫成獨立檔案，`--labels FILE` 寫出 Audacity 標籤軌。
沒有語音時結束碼為 1。

### pitch {#pitch}

```bash
kudio pitch take.wav --segments --labels voiced.txt
```

```text
median 128.5 Hz  ·  111.2-158.9 Hz (6.2 semitones)  ·  71% voiced
    1     0.560 -    2.460  (1.900s)
    2     2.960 -    5.250  (2.290s)
    3     5.970 -    7.450  (1.480s)
    4     8.160 -    9.550  (1.390s)
wrote 4 label(s) to voiced.txt
```

[`f0`][kudio.core.pitch.f0] 的摘要；`--segments` 加上有聲段落，`--labels` 寫出它們，`--fmin` 和
`--fmax` 設定搜尋範圍。沒有有聲音框時結束碼為 1。

## 轉換與編輯 {#converting-and-editing}

### convert {#convert}

```bash
kudio convert raw/day1/mic_a.wav mic_a.wav --rate 16000
kudio convert raw/ 16k/ --rate 16000 --recursive --lufs -23
```

```text
wrote mic_a.wav (16000 Hz, PCM_16)
3/3 file(s) -> 16k/
```

一個檔案，或加 `--recursive` 經由 [`convert_folder`][kudio.core.io.convert_folder] 處理整棵樹，
在輸出資料夾底下照原結構放：`--rate`、`--subtype`、`--lufs` 或 `--peak`、`--trim-db`。有檔案失敗
時結束碼為 1。

### normalize {#normalize}

```bash
kudio normalize hot.wav hot_r128.wav --lufs -23
kudio normalize take.wav loud.wav --lufs -6
```

```text
wrote hot_r128.wav normalised to -23.0 LUFS
wrote loud.wav normalised to -6.0 LUFS
warning: normalize_lufs: -6.0 LUFS puts the peak at 2.09 (6.4 dBFS over); the signal will clip when written to an integer format
```

`--lufs`（BS.1770 響度）或 `--peak`；保留聲道。增益讓樣本超過滿刻度時會回報，16 位元檔案會被
削峰，所以請寫 `--subtype FLOAT` 或把目標調低。

### trim {#trim}

```bash
kudio trim take.wav take_trim.wav --top-db 30
```

```text
trimmed [9216:153088] -> take_trim.wav (143872 samples)
```

用 [`trim_silence`][kudio.effects.silence.trim_silence] 修掉兩端比最響音框低 `--top-db`（30）以上
的靜音。

## 房間與通道 {#rooms-and-channels}

### room {#room}

```bash
kudio room --make room.wav --rt60 0.6 --drr 3 --seed 0
kudio room room.wav
kudio room take.wav --apply take_room.wav --rt60 0.6 --drr 3 --seed 0
```

```text
wrote room.wav (0.90s at 16000 Hz)
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
T20            0.603 s
T30            0.598 s
EDT            0.583 s
C50            +9.24 dB
C80           +12.11 dB
DRR            +3.00 dB
wrote take_room.wav — 0.6s room, DRR +3 dB, at 16000 Hz
T30 0.60 s (EDT 0.58 s)  ·  C50 +9.2 dB  ·  DRR +3.0 dB  ·  fit 1.000
```

`kudio room FILE` 用 [`rt60`][kudio.core.room.rt60] 量測脈衝響應，衰減看起來不像脈衝響應時結束碼
為 1；`--make` 用 [`rir`][kudio.core.room.rir] 建立房間；`--apply` 讓檔案經過由 `--rt60`、`--drr`
和 `--seed` 建出的合成房間。

### channel {#channel}

```bash
kudio channel take.wav phone.wav --telephone
kudio channel take.wav bursty.wav --loss 0.05 --burst 6 --seed 0
```

```text
wrote phone.wav
  · telephone (mu_law)
bandwidth     8000 Hz ->    3797 Hz
si_sdr       -4.54 dB against the original
wrote bursty.wav
  · 25 packet(s) lost in 6 run(s) of 4.2, 0.50s, silence
bandwidth     8000 Hz ->    8000 Hz
si_sdr      +23.84 dB against the original
```

`--telephone`、`--codec`、`--bits`、`--loss`、`--burst`、`--conceal` 和 `--packet-ms` 組成一條
鏈路，依鏈路本身的順序套用，指令會印出它做了什麼，包括掉了哪些封包。

## 建立含雜訊資料集 {#building-a-noisy-dataset}

### synth {#synth}

```bash
kudio synth --clean corpus/clean --noise corpus/noise --out corpus/mixed --snr -5 0 5 --seed 17
```

```text
synthesized 18 files -> corpus/mixed (seed 17)
INFO kudio.core.synth: ==== Audio synthesizing ====
INFO kudio.core.synth: Clean: 3, Noise: 2, SNR: [-5.0, 0.0, 5.0]
INFO kudio.core.synth: 18 noisy files from 3 SNRs, 2 noises and 1 room(s)
```

一次 [`Synthesizer`][kudio.core.synth.Synthesizer] 執行：`--mode`（預設 `inc`）、`--snr`、
`--rt60`、`--drr`、`--targets`、`--channel`、`--subtype`、`--rate` 和 `--manifest FILE`。它照
[建立含雜訊資料集](guide/synthesis.md)裡的保護規則清空 `--out`，並印出用的種子。

## 降噪 {#denoising}

### enhance {#enhance}

```bash
kudio enhance noisy.wav denoised.wav
kudio enhance noisy.wav denoised_omlsa.wav --method omlsa --noise quantile --set q=0.4
```

```text
log-MMSE + mcra noise estimate -> denoised.wav
  noise floor -28.8 -> -48.1 dBFS
OM-LSA + quantile noise estimate -> denoised_omlsa.wav
  noise floor -28.8 -> -51.5 dBFS
```

對一個檔案執行 [`spectral_enhance`][kudio.enhance.spectral.spectral_enhance]，或加 `--recursive`
經由 [`enhance_folder`][kudio.enhance.spectral.enhance_folder] 處理資料夾。`--set NAME=VALUE`
傳入方法參數。會印出處理前後的底噪。資料夾裡有檔案失敗時結束碼為 1。

### compare {#compare}

```bash
kudio compare noisy.wav --reference take.wav --methods logmmse wiener --noises all
```

```text
method + noise                   ms  floor dBFS    SI-SDR    dSNR   PESQ   STOI
-------------------------------------------------------------------------------
wiener + initial                 23       -45.7     15.50   10.62      —   0.53
logmmse + quantile               46       -48.6     14.95   10.01      —   0.53
logmmse + initial                58       -42.7     14.75    9.89      —   0.53
wiener + quantile                26       -53.2     14.75    9.79      —   0.52
logmmse + mcra                   57       -48.1     11.97    7.03      —   0.52
wiener + mcra                    35       -53.0     11.45    6.49      —   0.52
logmmse + minimum                83       -46.2      6.58    1.80      —   0.50
wiener + minimum                 54       -51.3      5.23    0.75      —   0.50
WARNING kudio.core.evaluator: PESQ unavailable: pip install "kudio[pesq]" (builds from source: needs a C compiler -- on Windows, the MSVC Build Tools)
```

[`compare_enhancers`][kudio.enhance.spectral.compare_enhancers]：所有方法，或 `--methods` 指定的，
配預設的估計方法、`--noise`，或 `--noises all`；有參考訊號時最好的排第一。`--write-to DIR` 寫出
每個輸出。

## 對齊兩段錄音 {#lining-up-two-recordings}

### align {#align}

```bash
kudio align take.wav processed.wav --metrics --write-to aligned/
kudio align take.wav corpus/noise/fan.wav
```

```text
lags by 128 samples (8.0 ms), correlation 0.970
overlap     10.000s of 10.000s / 10.008s

si_sdr         -2.16 dB  as the files sit
               11.97 dB  aligned

wrote 2 file(s) to aligned
leads by 11727 samples (732.9 ms), correlation 0.016
overlap     4.000s of 10.000s / 4.000s

these do not look like the same recording — the delay above does not mean anything
```

[`find_delay`][kudio.core.align.find_delay] 和 [`align`][kudio.core.align.align]：延遲、相關係數，
加 `--metrics` 時還有對齊前後的 SI-SDR；`--write-to DIR` 寫出對齊後的一對。兩者看起來不是同一段
錄音時結束碼為 1。

## 評整個測試集 {#scoring-a-test-set}

### score {#score}

```bash
kudio enhance testset/noisy testset/enhanced --recursive
kudio score testset/enhanced --reference testset/clean --baseline testset/noisy --metrics stoi si_sdr --groups "^(p\d+)_" --per-item scores.csv
```

```text
24/24 file(s) -> testset/enhanced
  noise floor down 9.7 dB on average
| Metric | Better | Baseline [95% CI] | Mean [95% CI] | Difference [95% CI] | p | p (Holm) | n | Not scored |
|---|---|---|---|---|---|---|---|---|
| STOI | higher | 0.720 [0.694, 0.740] | 0.732 [0.700, 0.758] | +0.012 [+0.005, +0.022] | 0.0181 | 0.0181 | 24 | 0 |
| SI-SDR (dB) | higher | 5.00 [4.99, 5.01] | 12.00 [11.90, 12.10] | +7.00 [+6.90, +7.10] | 0.0005 | 0.00098 | 24 | 0 |

Signature for the means and the baselines: kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|n:24|ci:kind=bootstrap_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
Signature for the differences: kudio:5.0.0|sr:16000|pair:aligned=no,cut=shorter,limit=0.5|si_sdr|stoi|n:24|ci:kind=paired_ci,method=bca,confidence=0.95,n_resamples=10000,seed=0,groups=12
References: kudio, taal2011stoi, leroux2019sdr, luo2019convtasnet
wrote 24 row(s) to scores.csv
```

[`score_dataset`][kudio.core.dataset_score.score_dataset] 依 id 配對檔案，來源可以是資料夾、
`id path` 清單或 CSV 檔，印出論文要的表：每個平均值和它以說話者為單位的 95% BCa 區間
（`--groups`：正規表示式、Kaldi `utt2spk` 或 CSV）、和 `--baseline` 的差異與它的區間及符號翻轉
p 值（跨指標做 Holm 調整）、[簽章](guide/metrics.md#signing-a-result)，以及 BibTeX 的鍵；`--bib FILE`
寫出條目，`--per-item FILE` 每句一列，`--format` 給 `latex` 或 `json`。沒有 `--reference` 時只算
不需參考訊號的指標。有語句在某一欄沒有分數時結束碼為 1。

## 音訊裝置 {#audio-devices}

### devices {#devices}

```bash
kudio devices
```

每個音訊裝置的編號、host API，以及輸入和輸出聲道數；`device=` 收的就是這個編號
（[錄音與即時檢視](guide/live-audio.md)）。

## 結束碼與 shell 測試 {#exit-codes-and-shell-tests}

| 指令 | 結束碼為 1 的情況 |
|---|---|
| `report` | 有任何發現 |
| `loudness --target` | 響度差超過 1 LU，或檔案從沒高過 -70 LUFS 門檻 |
| `vad` | 找不到語音 |
| `pitch` | 沒有有聲音框 |
| `align` | 相關係數低於 0.5 |
| `room FILE` | 衰減看起來不像脈衝響應 |
| `convert --recursive`、`enhance --recursive` | 有檔案失敗 |
| `score` | 有語句在某一欄沒有分數 |

結束碼 2 是錯誤，標準錯誤上一行訊息：用法錯誤、讀不了的檔案、函式庫拒絕的參數。其他情況結束碼
是 0。一項檢查在 shell 腳本裡就是一行；`||` 遇到結束碼 2 也會觸發，所以必要時請分開檢查 1 和 2：

```bash
kudio report hot.wav > /dev/null || echo "needs another take"
```
