# 房間與通道 {#rooms-and-channels}

雜訊是相加的，$y + n$。房間是摺積的，$y \ast h$：它不加任何東西，而是把每個聲音抹到之後的
幾百毫秒上。錄音經過的鏈路，kudio 稱為通道，兩者都不是：編解碼器做量化、轉換器丟掉位元、
網路掉封包。它們發生的順序固定：

```text
talker -> room -> + noise -> link -> file
```

這一頁製作並量測房間和鏈路；混入雜訊見[建立含雜訊資料集](synthesis.md)。這裡的函式都收單聲道
音訊，`(frames, channels)` 陣列會被 [`FeatureError`][kudio.exceptions.FeatureError] 拒絕。
範例共用三秒模擬語音的訊號：

```python
import numpy as np
import kudio

sr = 16000
t = np.arange(3 * sr) / sr
# a 140 Hz voice with a 4 Hz syllable rhythm, silent between the phrases
voice = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
y = 0.1 * voice * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
y[(t < 0.5) | ((t > 1.4) & (t < 1.9)) | (t > 2.6)] = 0.0
y += 0.003 * np.random.default_rng(0).standard_normal(t.size)   # a faint floor
y = y.astype(np.float32)
```

降噪器拿掉的是雜訊頻譜，而房間沒有加上任何頻譜，所以
[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] 對殘響沒有作用，掉掉的封包也
沒有東西能補回來。這裡的函式是在掌控之下製造這些劣化，並量測它們。

## 房間 {#rooms}

### 製作房間 {#making-one}

```python
ir = kudio.rir(sr, rt60=0.6, drr_db=6.0, seed=0)
print(len(ir) / sr)
print(kudio.rt60(ir, sr))
```

```text
0.9
T30 0.60 s (EDT 0.55 s)  ·  C50 +11.7 dB  ·  DRR +6.0 dB  ·  fit 1.000
```

[`rir`][kudio.core.room.rir] 用指數衰減的雜訊建出脈衝響應，這是擴散殘響尾巴的標準模型。
`rt60` 是包絡下降 60 dB 所需的時間；`drr_db` 是直達聲比殘響能量高多少：+10 dB 以上是近距離
麥克風，0 左右是一兩公尺遠。長度預設是 1.5 × `rt60`，同樣的 `seed` 建出同樣的房間。
`drr_db` 低於殘響尾巴本身在 2.5 ms 視窗內的能量時，會丟出 `FeatureError`。這個房間沒有早期
反射，不能拿來代替實測的響應；[`reverb`][kudio.effects.augment.reverb] 又更粗略。

### 把片段放進房間 {#putting-a-clip-in-it}

[`apply_rir`][kudio.core.room.apply_rir] 做摺積，把結果裁回輸入的長度（`trim=True`），並把
峰值放回輸入的位置（`normalise=True`）。它不收取樣率，所以實測的響應要用片段的取樣率載入；
除非想把聲音傳播時間當成延遲保留，否則從直達聲切起：

```python
wet = kudio.apply_rir(y, ir)
measured = ir[np.argmax(np.abs(ir)):]          # a recorded response, from its peak
print(len(wet) == len(y), np.max(np.abs(wet)) == np.max(np.abs(y)))
```

```text
True True
```

### 量測房間 {#measuring-one}

```python
noise = np.random.default_rng(1).standard_normal(len(y)).astype(np.float32)
noisy = kudio.add_noise_snr(y, noise, snr_db=5, seed=0)
print(kudio.rt60(noisy, sr))
room = kudio.rt60(kudio.rir(sr, rt60=1.0, seed=0)[:int(0.05 * sr)], sr)  # 50 ms of 1 s
print(f"T30 {room.t30:.2f} s, fit {room.fit:.3f}, reliable: {room.reliable()}")
```

```text
T30 2.62 s (EDT 18.47 s)  ·  C50 -11.2 dB  ·  DRR -20.3 dB  ·  fit 0.924  — but EDT and T30 disagree, so this may not be an impulse response
T30 0.19 s, fit 0.802, reliable: False
```

[`rt60`][kudio.core.room.rt60] 傳回 [`Reverberation`][kudio.core.room.Reverberation]。T20、T30
和 EDT 來自反向積分的衰減曲線[^schroeder]（[`schroeder_curve`][kudio.core.room.schroeder_curve]）：
從 −5 降到 −25 dB、−5 降到 −35 dB、0 降到 −10 dB 的時間，換算到 60 dB，也就是 ISO 3382-1
的評估範圍。[^iso3382] C50、C80 和 DRR 是峰值前後的能量比，`fit` 是 T30 擬合直線的 R²。量測
從第一個在峰值 20 dB 以內的樣本開始，所以聲音傳播時間不算在內。

反向積分會讓任何訊號都在衰減，所以先看 [`reliable()`][kudio.core.room.Reverberation.reliable]
再看數字：它要求 `fit` 至少 0.9，且 EDT 和 T30 相差不到 [`MAX_EDT_RATIO`][kudio.core.room.MAX_EDT_RATIO]
（2）倍。它看不到底噪：比雜訊高不到 45 dB 的響應根本沒有可引用的 T30；DRR 大約 +10 dB 以上時
沒有 EDT，`reliable()` 只看擬合。`rt60` 量的是整個頻帶；要某個八度音程時，用
`zero_phase=False` 濾波，再從峰值切一次。

![合成房間的 Schroeder 衰減曲線和它的 T30 擬合，以及同一個房間疊在底噪上](../images/degradation-schroeder-light.webp#only-light)
![合成房間的 Schroeder 衰減曲線和它的 T30 擬合，以及同一個房間疊在底噪上](../images/degradation-schroeder-dark.webp#only-dark)

## 通道 {#channels}

### 電話線 {#a-phone-line}

```python
phone = kudio.telephone(y, sr)
print(len(phone) == len(y))
for problem in kudio.audio_report(phone, sr).problems():
    print(problem)
```

```text
True
content stops at 3758 Hz, although a 16000 Hz file holds up to 8000 Hz — it was very likely upsampled from a lower rate, or passed through a codec or a filter that cuts there, and the band above is empty
```

[`telephone`][kudio.effects.channel.telephone] 在原取樣率下限制頻帶到 300–3400 Hz
（[`TELEPHONE_BAND`][kudio.effects.channel.TELEPHONE_BAND]），重新取樣到 8 kHz，經過 G.711[^g711]
mu-law 壓伸，再以原來的取樣率和長度回來。

### 壓伸與位元深度 {#companding-and-bit-depth}

```python
full = y / np.max(np.abs(y))
for level in (0, -12, -24):
    x = (full * 10 ** (level / 20)).astype(np.float32)
    print(f"{level:3} dBFS   mu-law {kudio.snr(x, kudio.mu_law(x)):.1f}   "
          f"A-law {kudio.snr(x, kudio.a_law(x)):.1f}   "
          f"8-bit linear {kudio.snr(x, kudio.bit_depth(x, 8)):.1f}")
```

```text
  0 dBFS   mu-law 37.9   A-law 38.1   8-bit linear 40.2
-12 dBFS   mu-law 37.7   A-law 38.0   8-bit linear 28.2
-24 dBFS   mu-law 36.6   A-law 36.6   8-bit linear 18.5
```

[`mu_law`][kudio.effects.channel.mu_law] 和 [`a_law`][kudio.effects.channel.a_law] 在連續的壓伸
曲線上量化成 8 位元碼字，所以量化雜訊會跟著訊號一起變小；[`bit_depth`][kudio.effects.channel.bit_depth]
則是線性地用掉這些位元。它們用的是曲線本身，不是 G.711 的分段表，夠資料擴增用，但不是位元
精確的解碼器。`bit_depth(dither=True)` 加上兩階寬的 TPDF 抖動；請同時傳 `seed`。

### 封包遺失 {#packet-loss}

```python
lossy, gaps = kudio.dropouts(y, sr, loss=0.05, seed=0, return_spans=True)
print(len(gaps), [f"{a:.2f}-{b:.2f} s" for a, b in gaps[:3]])
```

```text
8 ['0.04-0.06 s', '0.12-0.14 s', '0.22-0.24 s']
```

[`dropouts`][kudio.effects.channel.dropouts] 把片段切成 `packet_ms`（20 ms，
[`PACKET_MS`][kudio.effects.channel.PACKET_MS]，G.711 走 RTP[^rtp] 時的封包）長的封包，掉掉
`round(loss * packets)` 個：三秒是 150 個封包，所以掉 8 個。預設的 `conceal='silence'` 讓洞
空著；`conceal='hold'` 重複每個洞前面的封包。在波形指標上 `hold` 會不會比較好，取決於基頻週期
和封包長度怎麼對上，所以 SNR 或 SI-SDR 沒辦法替你在兩者之間做選擇。

`burst=` 設定平均連續遺失的封包數，長度呈幾何分布，和 Gilbert 模型一樣，[^gilbert] 每段之間至少
隔一個封包；遺失率超過一半時可能做不到，`dropouts` 會丟出 `FeatureError`。預設的 `burst=None`
讓封包各自獨立遺失，這樣平均每段已經有 [`independent_burst`][kudio.effects.channel.independent_burst]
個封包（5% 遺失率時是 1.05）；`burst=1` 則是每段剛好一個。

![30 秒、5% 的封包遺失，分別是獨立遺失和平均 6 個一段](../images/degradation-loss-light.webp#only-light)
![30 秒、5% 的封包遺失，分別是獨立遺失和平均 6 個一段](../images/degradation-loss-dark.webp#only-dark)

### 一次套用整條鏈路 {#the-whole-link-at-once}

```python
right = kudio.apply_channel(y, sr, {"preset": "telephone", "loss": 0.1}, seed=0)
wrong = kudio.telephone(kudio.dropouts(y, sr, loss=0.1, seed=0), sr)
print(np.count_nonzero(right == 0.0), np.count_nonzero(wrong == 0.0))
print(kudio.channel_tag({"preset": "voip", "loss": 0.1}))
```

```text
4800 0
mu_law+loss10%x4-hold
```

[`apply_channel`][kudio.effects.channel.apply_channel] 照鏈路本身的順序執行：頻帶與取樣率、
編解碼器、轉換器，最後才是網路，所以掉掉的封包維持完全靜音；先掉封包的話，濾波器會把前後的
語音抹進每個洞裡。規格可以是 [`CHANNEL_PRESETS`][kudio.effects.channel.CHANNEL_PRESETS] 的預設
（`telephone`、`telephone_a`、`mu_law`、`a_law`、`voip`、`none`）、由
[`CHANNEL_DEFAULTS`][kudio.effects.channel.CHANNEL_DEFAULTS] 的鍵組成的 dict，或預設加上覆寫。
[`channel_spec`][kudio.effects.channel.channel_spec] 在任何音訊通過之前，就拒絕不認得的設定、
0 到 1 以外的遺失率、不在 2 到 64 的位元數、小於 1 的 `burst`，以及不是單一有限數值的東西。
[`Synthesizer`][kudio.core.synth.Synthesizer] 以 `channel=` 收同樣的規格。

## 從命令列 {#from-the-command-line}

```bash
kudio room ir.wav                                        # exit 1 if not reliable()
kudio room --make room.wav --rt60 0.6 --drr 6 --seed 0
kudio room speech.wav --apply wet.wav --rt60 0.6 --seed 0
kudio channel in.wav out.wav --telephone
kudio channel in.wav out.wav --loss 0.05 --burst 6 --conceal hold --seed 0
```

參數在 [API 參考](../reference/degradation.md)。

[^schroeder]: M. R. Schroeder, "New method of measuring reverberation time", J. Acoust. Soc. Am. 37(3), pp. 409-412, 1965, doi:10.1121/1.1909343.
[^iso3382]: ISO 3382-1:2009, "Acoustics — Measurement of room acoustic parameters — Part 1: Performance spaces".
[^g711]: ITU-T Recommendation G.711 (11/1988), "Pulse code modulation (PCM) of voice frequencies".
[^rtp]: H. Schulzrinne and S. Casner, "RTP profile for audio and video conferences with minimal control", RFC 3551, July 2003, Sec. 4.2, doi:10.17487/RFC3551.
[^gilbert]: E. N. Gilbert, "Capacity of a burst-noise channel", Bell Syst. Tech. J. 39(5), pp. 1253-1265, 1960, doi:10.1002/j.1538-7305.1960.tb03959.x.
