# 檢查錄音 {#inspecting-a-recording}

現場的錄音沒有乾淨的版本可以比對。這些函式只看錄音本身：

| 問題 | 用 | 需要 |
|---|---|---|
| 技術上有沒有毛病？ | [`audio_report`][kudio.core.report.audio_report] | 核心安裝 |
| 有沒有突出的嗡聲或尖鳴？ | [`tonal_lines`][kudio.core.lines.tonal_lines] | 核心安裝 |
| 聽起來多大聲？ | [`loudness`][kudio.core.loudness.loudness] 和旁邊的函式 | 核心安裝 |
| 殘響多重？ | [`srmr`][kudio.core.srmr.srmr] | 核心安裝 |
| 語音比雜訊高多少？ | [`wada_snr`][kudio.core.wada.wada_snr] | 核心安裝 |
| 聽的人會怎麼說？ | [`dnsmos`][kudio.core.dnsmos.dnsmos] | `kudio[dnsmos]` 和 DNS-Challenge 的權重 |

範例共用一段錄音：一個 140 Hz 的聲音在嘶聲上時有時無。

```python
import numpy as np
import kudio

sr = 16000
rng = np.random.default_rng(0)
t = np.arange(20 * sr) / sr
voice = sum(np.cos(2 * np.pi * 140 * k * t) / k for k in range(1, 55))
words = np.sin(2 * np.pi * 0.7 * t) > -0.3          # talk, pause, talk, ...
take = 0.05 * voice * words + 0.002 * rng.standard_normal(t.size)
```

## 不需參考訊號的健康檢查 {#a-health-check-with-no-reference}

```python
report = kudio.audio_report(take, sr)
print(report)
print(report.problems())
```

```text
20.00s @ 16000 Hz · peak -12.6 dBFS · -29.2 LUFS · SNR~28 dB · bandwidth 8000 Hz
[]
```

[`audio_report`][kudio.core.report.audio_report] 收單聲道音訊，以 30 ms 音框、15 ms
跳距分析；[`problems()`][kudio.core.report.AudioReport.problems] 把值得處理的發現寫成句子，
空清單就是好消息。立體聲檔案請一個聲道一個聲道檢查。

| 欄位 | 量的是 | 何時列為問題 |
|---|---|---|
| `peak_dbfs` | 最大的樣本，單位 dBFS | 超過 0、超過 -0.5 或低於 -30 |
| `clipped_samples`、`clipped_ratio` | 在滿刻度上連續平坦、長度至少 `clip_run`（3）的樣本 | 有就列 |
| `full_scale_samples` | 達到滿刻度 0.999 以上的樣本，不論是否連續 | 不列 |
| `dc_offset` | 樣本平均值 | 大小超過 0.01 |
| `silence_ratio` | 比第 95 百分位音框低 40 dB 以上的音框比例 | 超過 0.6 |
| `noise_floor_dbfs` | 第 10 百分位音框的位準 | 不列 |
| `estimated_snr_db` | 第 95 百分位音框位準減第 10 百分位 | 低於 10 dB |
| `bandwidth_hz` | 在最強頻率 50 dB 以內的最高頻率 | 不列 |
| `band_limited`、`source_rates`、`mirrored`、`band_edge_hz` | 上方頻帶全空的邊緣，或線性內插留下的鏡像 | 找到就列 |
| `sample_step`、`grid_bits` | 所有樣本落在的格點 | 比 16 位元粗 |
| `lufs`、`rms_dbfs` | 整合響度（短於 0.4 s 是 `nan`）和單純的 RMS | 不列 |
| `non_finite_samples` | NaN 或 inf 的樣本 | 有就列，而且只列這一項 |

SNR 估計是從最安靜的一成音框讀出底噪，所以需要含雜訊的停頓：沒有停頓時會讀到大約 1 dB，
意思是不知道，不是很差。含 NaN 或 inf 的片段照樣量，唯一的發現就是個數。全部通過只是
技術上的結論；聽起來如何要看 [DNSMOS](#dnsmos-what-a-listener-would-say)。

### 削峰與頻寬受限 {#clipping-and-band-limits}

```python
hot = np.clip(8 * take, -1.0, 1.0)                         # written too hot
narrow = kudio.resample(kudio.resample(take, sr, 8000), 8000, sr)   # 8 kHz audio
for clip in (hot, narrow):
    for problem in kudio.audio_report(clip, sr).problems():
        print(problem)
```

```text
3321 samples clipped (1.04% of the clip) — the recording was already too hot; no processing gets that back
content stops at 3816 Hz, although a 16000 Hz file holds up to 8000 Hz — it was very likely upsampled from 8000 Hz (a codec that cuts there looks the same), and the band above is empty
```

樣本要在滿刻度 0.999 以上、連續 `clip_run`（3）個以上，才算被削峰；超過 1.0 的浮點音訊
其實還完整，只會列為峰值問題。

![錄音和它升取樣後的版本的平均頻譜，標出報告在 4 kHz 比較的兩個頻帶](../images/inspection-bandwidth-light.webp#only-light)
![錄音和它升取樣後的版本的平均頻譜，標出報告在 4 kHz 比較的兩個頻帶](../images/inspection-bandwidth-dark.webp#only-dark)

升取樣過的檔案會有一道邊緣，上方頻帶是空的：比邊緣下方的頻帶低 20 dB，一路到 Nyquist
都低 40 dB；`source_rates` 列出符合的取樣率。內容比底噪高不到 40 dB、或邊緣在 Nyquist
的 0.9 以上時抓不到（[驗證過的情況](../validation.md#upsampling-and-band-limits-in-audio_report)）。
樣本格點比 16 位元粗時，會給出 `grid_bits` 並列為問題。

### 嗡聲和其他窄頻線 {#hum-and-other-lines}

[`tonal_lines`][kudio.core.lines.tonal_lines] 找出平均頻譜裡比周圍滑動中位數高 20 dB 以上的
峰，再把落在同一頻率整數倍上的峰歸成一組梳狀結構，市電嗡聲就是這樣：嘶聲加上 50 Hz 嗡聲
和 1 kHz 尖鳴時，它回報 `6 lines, the strongest at 49.9 Hz, 35 dB above its
neighbourhood; 50.00 Hz comb, 5 lines (harmonics 1-5)`。它只說哪些頻率突出，不說為什麼；
拉長的母音也是梳狀結構。

## 聽起來有多大聲 {#how-loud-it-sounds}

[`loudness`][kudio.core.loudness.loudness] 依 ITU-R BS.1770-4[^bs1770] 給出有門檻的整合響度，
單位 LUFS：K 加權、每 100 ms 取 400 ms 區塊的均方值、-70 LUFS 絕對門檻和低 10 LU 的相對
門檻，最後平均。門檻把停頓排除在外：錄音後面接十秒房間底噪，RMS 從 -29.2 降到
-30.9 dBFS，響度仍是 -29.2 LUFS。多聲道以聲道在後的方式傳入，最多五個，所以立體聲檔案要用
`mono=False` 載入。短於一個區塊的片段會丟出指向 `block_size=` 的 `FeatureError`。

### 正規化與對齊響度 {#normalising-and-matching}

```python
loud = kudio.normalize_lufs(take, sr, lufs=-14.0)
noisy = take + 0.02 * rng.standard_normal(t.size)
matched = kudio.match_loudness(kudio.spectral_enhance(noisy, sr), noisy, sr)
print(f"{kudio.loudness(loud, sr):.1f} LUFS, peak {np.max(np.abs(loud)):.2f}")
print(f"{kudio.loudness(noisy, sr):.1f} and {kudio.loudness(matched, sr):.1f} LUFS")
```

```text
-14.0 LUFS, peak 1.34
-27.5 and -27.5 LUFS
```

[`normalize_lufs`][kudio.core.loudness.normalize_lufs] 只套一個增益，預設到 -23 LUFS
（EBU R 128[^ebu]），而且不做限幅：超過滿刻度的樣本會照樣超過 1.0，並發出 `UserWarning`。
[`match_loudness`][kudio.core.loudness.match_loudness] 把一段片段調到另一段的響度，做
A/B 比較前要先這樣做。

## 響度隨時間的變化 {#loudness-over-time}

```python
drift = take.copy()
drift[8 * sr:13 * sr] *= 10 ** (-12 / 20)          # five seconds facing away
curve = kudio.loudness_over_time(drift, sr)
print(curve)
print(curve.spans_below(curve.max - 6, min_seconds=1.0))
print(f"{kudio.loudness_range(drift, sr):.1f} LU")
```

```text
-29.7 LUFS at 16.30s, -42.1 LUFS at 11.00s (12.4 LU apart)
[(9.1, 11.9)]
12.0 LU
```

![錄音在 8 到 13 秒降低 12 dB 時的瞬時響度和短期響度](../images/inspection-loudness-light.webp#only-light)
![錄音在 8 到 13 秒降低 12 dB 時的瞬時響度和短期響度](../images/inspection-loudness-dark.webp#only-dark)

[`loudness_over_time`][kudio.core.loudness.loudness_over_time] 每 100 ms 量一個 3 s 的視窗，
或用 [`MOMENTARY`][kudio.core.loudness.MOMENTARY] 量 400 ms；`times` 是視窗中心，靜音是
`-inf`。[`loudness_range`][kudio.core.loudness.loudness_range] 是 EBU Tech 3342 定義的值。

## 真峰值 {#true-peak}

```python
tone = np.sin(2 * np.pi * (sr / 4) * np.arange(sr) / sr + np.pi / 4)  # samples at ±0.707
tricky = kudio.fade(tone, sr, 0.01, 0.01)
peak = 20 * np.log10(np.max(np.abs(tricky)))
print(f"samples {peak:.2f} dBFS, true peak {kudio.true_peak(tricky, sr):+.2f} dBTP")
```

```text
samples -3.01 dBFS, true peak +0.02 dBTP
```

[`true_peak`][kudio.core.loudness.true_peak] 過取樣到至少 192 kHz（BS.1770-4 附錄 2）。
EBU R 128 容許最多 -1 dBTP，而 `normalize_lufs` 只對樣本發警告，所以正規化之後要再量真峰值。

## 殘響有多重：SRMR {#how-reverberant-srmr}

```python
syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)      # four a second
talk = 0.05 * voice * words * syllables + 0.002 * rng.standard_normal(t.size)
for rt60 in (0.3, 0.6, 1.0):
    wet = kudio.apply_rir(talk, kudio.rir(sr, rt60=rt60, seed=0))
    print(f"RT60 {rt60:.1f} s  SRMR {kudio.srmr(wet, sr):.2f}")
```

```text
RT60 0.3 s  SRMR 10.08
RT60 0.6 s  SRMR 9.40
RT60 1.0 s  SRMR 8.11
```

[`srmr`][kudio.core.srmr.srmr] 是語音對殘響的調變能量比[^srmr]，語音越乾越高，支援 8 和
16 kHz。它沒有固定範圍，所以同一段錄音只跟它自己在別處的版本比；`norm=True` 是同時有雜訊
和殘響時用的變體。

## 雜訊有多大：WADA-SNR {#how-noisy-wada-snr}

```python
speech = 0.05 * voice * words * syllables
noise = np.random.default_rng(2).standard_normal(t.size)
for snr_db in (0, 10, 20):
    noisy = kudio.add_noise_snr(speech, noise, snr_db=snr_db, seed=0)
    print(f"{snr_db:2d} dB  WADA-SNR {kudio.wada_snr(noisy, sr):6.2f}")
```

```text
 0 dB  WADA-SNR   0.55
10 dB  WADA-SNR  10.57
20 dB  WADA-SNR  25.48
```

[`wada_snr`][kudio.core.wada.wada_snr] 從樣本分布有多尖讀出 SNR。[^wada] 20 dB 以上時語音
本身的影響比雜訊大，所以讀到很高的值就當作「乾淨」。短於 1 s，或停頓是數位靜音的片段，
得到 nan。

## DNSMOS：聽的人會怎麼說 {#dnsmos-what-a-listener-would-say}

[`dnsmos`][kudio.core.dnsmos.dnsmos] 照 DNS-Challenge 的 `dnsmos_local.py` 執行 Microsoft 的
DNSMOS P.835 模型[^dnsmos]，傳回 1 到 5 分的 SIG、BAK 和 OVRL；`personalized=True` 執行
個人化模型[^dns2023]，會再扣第二個說話者的分數；[`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808]
執行 P.808 模型[^p808]，只有一個總分。要裝 `kudio[dnsmos]`，再 clone
[DNS-Challenge](https://github.com/microsoft/DNS-Challenge) 取得權重：

```python
score = kudio.dnsmos(take, sr, model_dir="DNS-Challenge/DNSMOS")
print(score.summary())
```

音訊以 16 kHz、每 1 s 一個 9.01 s 的視窗評分，較短的片段會重複到填滿一個視窗；`session=`
讓整個資料夾共用一個已開啟的模型。`dnsmos_p808` 依照腳本釘住的 librosa 0.8.1，
`librosa="0.9"` 則用較新的頻譜圖（[指標驗證](../validation.md#dnsmos)）。

## 從命令列 {#from-the-command-line}

`kudio report take.wav` 找到問題時結束碼是 1，`kudio loudness take.wav --target -23` 在沒達到
目標時是 1；`kudio score` 給出整個測試集的平均 SRMR、WADA-SNR 或 DNSMOS
（[命令列](../cli.md#score)）。

[^srmr]: T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech", IEEE Trans. Audio, Speech, Lang. Process. 18(7), pp. 1766-1774, 2010, doi:10.1109/TASL.2010.2052247. The normalised variant: J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive intelligibility metric for noisy and reverberant speech", Proc. IWAENC 2014, pp. 55-59, doi:10.1109/IWAENC.2014.6953337.
[^bs1770]: Recommendation ITU-R BS.1770-4 (10/2015), "Algorithms to measure audio programme loudness and true-peak audio level". Annex 1 is the loudness measurement and Annex 2 the true-peak meter.
[^ebu]: EBU R 128 (2023), "Loudness normalisation and permitted maximum level of audio signals"; EBU Tech 3341 (November 2023), "Loudness metering: 'EBU Mode' metering to supplement EBU R 128 loudness normalization"; EBU Tech 3342 (November 2023), "Loudness range: a measure to supplement EBU R 128 loudness normalization".
[^dnsmos]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2022, pp. 886-890, doi:10.1109/ICASSP43922.2022.9746108. It was trained on ratings collected to ITU-T Recommendation P.835 (11/2003), "Subjective test methodology for evaluating speech communication systems that include noise suppression algorithm".
[^wada]: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis", Proc. Interspeech 2008, pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
[^p808]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2021, pp. 6493-6497, doi:10.1109/ICASSP39728.2021.9414878. It was trained on ratings collected with the crowdsourcing method of ITU-T Recommendation P.808 (06/2018), "Subjective evaluation of speech quality with a crowdsourcing approach".
[^dns2023]: H. Dubey, A. Aazami, V. Gopal, B. Naderi, S. Braun, R. Cutler, A. Ju, M. Zohourian, M. Tang, M. Golestaneh and R. Aichner, "ICASSP 2023 Deep Noise Suppression Challenge", IEEE Open Journal of Signal Processing 5, pp. 725-737, 2024, doi:10.1109/OJSP.2024.3378602; first posted as arXiv:2303.11510, 2023.
