# 入門 {#getting-started}

安裝 kudio、把一段有雜訊的錄音處理成降噪後的檔案，以及出問題時 kudio 會丟出什麼、警告
什麼、記錄什麼。

## 安裝 {#installing}

```bash
pip install kudio
```

核心安裝是 numpy、scipy、soundfile、soxr、PyYAML 和 tqdm：檔案、特徵、效果、響度、
降噪、資料集合成，以及 PESQ 和 DNSMOS 以外的所有指標。kudio 需要 Python 3.9 以上；CI
在 Linux 和 Windows 上測 3.9、3.11 和 3.13。`import kudio` 不會匯入任何選用套件，所以
缺了哪個選用套件組，要到呼叫需要它的函式時才會發現。

### 選用套件組 {#extras}

方括號要加引號，不然 zsh 會把它當成檔名萬用字元：

```bash
pip install "kudio[audio,viz]"
```

| 套件組 | 會安裝 | 用途 |
|---|---|---|
| `audio` | sounddevice | [`record`][kudio.core.stream.record]、[`StreamRecorder`][kudio.core.stream.StreamRecorder]、[`play_audio`][kudio.core.stream.play_audio]、[`list_devices`][kudio.util.check.list_devices]；Linux 上要有系統的 PortAudio，Debian 和 Ubuntu 是 `libportaudio2` |
| `pesq` | pesq | PESQ（ITU-T P.862 和 P.862.2），[`composite`][kudio.core.quality.composite] 裡也用到；PyPI 上只有原始碼，所以要有 C 編譯器，Windows 上是 MSVC Build Tools |
| `dnsmos` | onnxruntime | [`dnsmos`][kudio.core.dnsmos.dnsmos] 和 [`dnsmos_p808`][kudio.core.dnsmos.dnsmos_p808]；權重是 Microsoft 的，把 DNS-Challenge 的 repository clone 下來，再把裡面的 DNSMOS 資料夾傳給 `model_dir=`，或設 `KUDIO_DNSMOS_DIR` |
| `viz` | matplotlib | [`kudio.util.visual`][kudio.util.visual] 裡的畫圖工具 |
| `data` | pandas、openpyxl | [`AudioEvaluate.get_df`][kudio.core.evaluator.AudioEvaluate.get_df]、[`per_file`][kudio.core.evaluator.AudioEvaluate.per_file]、[`data2xlsx`][kudio.util.conv.data2xlsx] |
| `compressed` | audioread | m4a、aac、webm，以及 ffmpeg 能解碼的其他格式（[音訊檔案](guide/audio-files.md#formats-soundfile-does-not-read)）；ffmpeg 要在 PATH 上 |
| `eval` | 不裝任何東西 | 空的套件組：STOI、ESTOI 和 SDR 只用核心安裝就能算 |
| `all` | `audio`、`eval`、`dnsmos`、`viz`、`data`、`compressed` | `pesq` 以外的所有執行期套件組，`pesq` 要編譯器 |

`[dev]`、`[check]` 和 `[docs]` 是開發 kudio 本身用的。

## 第一次使用 {#a-first-session}

一段有雜訊的錄音：檢查、降噪、評分、存檔。先在空資料夾裡執行下面這段：它會寫出
`clean.wav`，兩段有聲的片段，前後有停頓；以及 `noisy.wav`，同樣的訊號混上 2 dB SNR
的白雜訊。

??? example "產生 `clean.wav` 和 `noisy.wav`"

    ```python
    import numpy as np

    import kudio

    sr = 16000
    t = np.arange(int(4.5 * sr)) / sr


    def syllables(start, stop, f0):
        """Harmonics of f0 under a 4 Hz syllable envelope, between two times."""
        inside = (t >= start) & (t < stop)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * (t - start))
        return np.where(inside, tone * envelope / 5, 0.0)


    clean = (syllables(0.5, 1.75, 120) + syllables(2.5, 4.0, 150)).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    noisy = kudio.add_noise_snr(clean, noise, snr_db=2, seed=0)

    kudio.save_wave("clean.wav", clean, sr)
    kudio.save_wave("noisy.wav", noisy, sr)
    ```

### 載入並檢查 {#load-and-check}

```python
import kudio

y, sr = kudio.file_load("noisy.wav")
print(kudio.audio_info("noisy.wav"))
report = kudio.audio_report(y, sr)
print(report)
for problem in report.problems():
    print(problem)
```

```text
noisy.wav: 4.500s, 16000 Hz, 1ch, PCM_16, peak 0.5430
4.50s @ 16000 Hz · peak -5.3 dBFS · -18.4 LUFS · SNR~9 dB · bandwidth 8000 Hz
estimated SNR around 9 dB — noise close to the content gives that, and so does a steady sound such as hum or a running machine, whose loud and quiet frames hardly differ; the estimate cannot tell the two apart
```

[`file_load`][kudio.core.io.file_load] 傳回 float32，取樣率是檔案本身的，或 `sr=`
指定的；多聲道會混成單聲道，除非 `mono=False`，那時形狀是 `(frames, channels)`。
[`audio_report`][kudio.core.report.audio_report] 不需要乾淨的參考訊號。它的 SNR 是拿
響的 30 ms 音框和安靜的比，所以 2 dB 混出來的檔案會讀到大約 9 dB：用來找出吵的檔案，
不是用來量混音的 SNR。

### 降噪 {#denoise}

```python
denoised = kudio.spectral_enhance(y, sr)
```

[`spectral_enhance`][kudio.enhance.spectral.spectral_enhance] 預設是 MCRA 雜訊估計加
log-MMSE 增益，傳回的樣本數和輸入一樣。

![noisy.wav 和 denoised.wav 在 4.5 秒內的音框位準，標出 noisy 位準的第 95 和第 10 百分位，相差 8.8 dB](images/start-levels-light.webp#only-light)
![noisy.wav 和 denoised.wav 在 4.5 秒內的音框位準，標出 noisy 位準的第 95 和第 10 百分位，相差 8.8 dB](images/start-levels-dark.webp#only-dark)

雜訊底從 -23.6 降到 -42.8 dBFS，音節的峰值只降了大約 1 dB。
[降噪](guide/enhancement.md)有其他的增益規則和雜訊估計方法。

### 跟乾淨訊號比對評分 {#score-against-the-clean-signal}

```python
ref, _ = kudio.file_load("clean.wav")
before = kudio.score(ref, y, sr, ["snr", "si_sdr", "segsnr"])
after = kudio.score(ref, denoised, sr, ["snr", "si_sdr", "segsnr"])
for name in before:
    print(f"{name:7s} {before[name]:6.2f} -> {after[name]:6.2f} dB")
```

```text
snr       2.00 ->  11.37 dB
si_sdr    1.99 ->  11.04 dB
segsnr   -3.56 ->   1.01 dB
```

兩次呼叫都會警告參考訊號 280 個音框裡有 105 個是數位靜音，segmental SNR 把這些音框算成
它的下限 -10 dB。[`score`][kudio.core.evaluator.score] 不給清單時會算 `snr`、
`si_sdr`、`segsnr`、`sdi`、`stoi`、`estoi` 和 `sdr`，裝了 `[pesq]` 且取樣率是 8 或
16 kHz 時再加上 `pesq`。它拿第 *i* 個樣本和第 *i* 個樣本比，所以可能錯開的一對要先
[`align`][kudio.core.align.align]：見[品質評估](guide/metrics.md)。

### 存檔 {#save}

```python
kudio.save_wave("denoised.wav", denoised, sr)
print(kudio.audio_info("denoised.wav"))
```

```text
denoised.wav: 4.500s, 16000 Hz, 1ch, PCM_16, peak 0.3796
```

[`save_wave`][kudio.core.io.save_wave] 預設寫 16 位元 PCM，`subtype=` 可以改；整數格式
會把超過 ±1.0 的部分削掉，NaN 或 inf 則直接拒絕。在 shell 裡，這一段就是
`kudio report noisy.wav` 和 `kudio enhance noisy.wav denoised.wav`（[命令列](cli.md)）。

## 錯誤 {#errors}

kudio 自己的例外都繼承 [`KudioError`][kudio.exceptions.KudioError]：

| 類別 | 什麼時候丟出 |
|---|---|
| [`AudioIOError`][kudio.exceptions.AudioIOError] | 檔案讀不了或寫不了；[`UnsupportedFormatError`][kudio.exceptions.UnsupportedFormatError] 是它的子類別，kudio 本身不會丟 |
| [`FeatureError`][kudio.exceptions.FeatureError] | 轉換或量測拒絕輸入：形狀不對、太短、方法或參數不認得；它也是 `ValueError` |
| [`DeviceError`][kudio.exceptions.DeviceError] | 沒有能錄音的輸入裝置，或 sounddevice 找不到 PortAudio 函式庫 |
| [`SynthesisError`][kudio.exceptions.SynthesisError] | [`Synthesizer`][kudio.core.synth.Synthesizer] 不肯清空放著輸入檔的資料夾、工作目錄、家目錄或檔案系統根目錄，或遇到沒辦法設定 SNR 的靜音檔 |
| [`DependencyError`][kudio.exceptions.DependencyError] | 缺選用套件；它也是 `ImportError`，訊息裡寫著要裝哪個套件組 |

`KudioError` 以外的情況：

- 參數檢查丟 `ValueError` 或 `TypeError`：不認得的指標、給指標的立體聲對、取樣率 0。
  輸入不存在是 `FileNotFoundError`。
- 每個指標都會拒絕 NaN、inf 和 int64 樣本，丟出以指標名稱開頭的 `ValueError`；沒有東西
  可算時，例如空的一對或靜音的參考訊號，給 nan 並發一次 `UserWarning`。
- 緩衝區等超過 `timeout=` 時丟 `TimeoutError`。
- [`play_audio`][kudio.core.stream.play_audio] 播不了時不丟例外，而是傳回一段說明；播成功
  傳回 `None`。

## 警告與記錄 {#warnings-and-logging}

`UserWarning` 表示音訊或結果本身有問題：檔案裡有 NaN、片段被推過滿刻度、兩個訊號被裁到
相同長度。不開 logging 也看得到。記錄裡是例行的事，寫在 `kudio` 底下的 logger，程式
沒設定 `logging` 之前什麼都不會印出來；`logging.basicConfig(level=logging.WARNING)`
之後，向 [`check_file`][kudio.core.io.check_file] 要一個已被佔用的檔名，會記下
`WARNING:kudio.core.io:Existing file will be overwritten: noisy.wav`。

`kudio` 指令的記錄等級是 `INFO`，加 `-v` 是 `DEBUG`。`TQDM_DISABLE=1` 會關掉批次工具的
進度條（tqdm 4.66 以上）。

刻意不出聲的地方：整數格式會削掉超過 ±1.0 的部分，混音可能過熱時請寫
`subtype="FLOAT"`；[`pesq`][kudio.core.evaluator.pesq] 在 16 kHz 預設給寬頻 MOS-LQO，
`mode=` 和 `scale=` 可以改；所有用到亂數的地方都收 `seed=`，預設 `None`，每次執行重新抽。

[使用指南](guide/index.md)依工作挑頁面，[API 參考](reference/index.md)有每個參數。
