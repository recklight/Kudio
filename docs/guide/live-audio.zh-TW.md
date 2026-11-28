# 錄音與即時檢視 {#recording-and-live-views}

還在進來的音訊：從麥克風或網路上的裝置取得，錄的同時畫出來或降噪，以及播放。

## 需要 audio 套件組的部分 {#what-needs-the-audio-extra}

```bash
pip install "kudio[audio]"     # sounddevice
```

kudio 透過 PortAudio 的綁定 sounddevice 存取硬體，只在需要時才匯入。它在 Windows 和 macOS
的 wheel 自帶 PortAudio；Linux 上用系統的，Debian 和 Ubuntu 是 `libportaudio2`。

| 後端 | 用到的地方 |
|---|---|
| sounddevice | [`record`][kudio.core.stream.record]、[`list_devices`][kudio.util.check.list_devices]、[`play_audio`][kudio.core.stream.play_audio]、[`StreamRecorder`][kudio.core.stream.StreamRecorder]、[`Recorder`][kudio.core.stream.Recorder]、[`LocalStreamReader`][kudio.core.stream.LocalStreamReader]、[`RemoteStreamReader`][kudio.core.stream.RemoteStreamReader]、[`CheckDevice`][kudio.util.check.CheckDevice] |
| 不需要 | [`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream]、[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer]、[`AudioBuffer`][kudio.core.buffer.AudioBuffer]、[`FetchBuffer`][kudio.core.buffer.FetchBuffer] |

沒裝這個套件組時，這些會丟出寫著 `kudio[audio]` 的 [`DependencyError`][kudio.exceptions.DependencyError]；
Linux 上裝了 sounddevice 卻沒有 PortAudio 函式庫時，丟出寫著系統套件名稱的
[`DeviceError`][kudio.exceptions.DeviceError]。

## 用 `record` 錄固定長度 {#a-fixed-length-take-with-record}

```python
import kudio

y = kudio.record(seconds=3, sr=16000)          # blocks for 3 s
kudio.save_wave("take.wav", y, 16000)
```

[`record`][kudio.core.stream.record] 傳回 float32、`int(seconds * sr)` 個樣本，單聲道是 `(n,)`，
否則是 `(n, channels)`。裝置不接受這個取樣率時，丟出寫明裝置和取樣率的 `DeviceError`。

## 錄到你喊停：`StreamRecorder` {#recording-until-you-stop-streamrecorder}

```python
import time

with kudio.StreamRecorder(sr=16000) as rec:
    print(f"{rec.sr} Hz, {rec.channels} channel(s)")   # what the device agreed to
    while rec.elapsed() < 5.0:                         # or until someone clicks Stop
        print(f"\r{rec.level_db():6.1f} dBFS", end="")
        time.sleep(0.1)
y = rec.stop()                                         # float32, (n,) for mono
kudio.save_wave("take.wav", y, rec.sr)
```

[`StreamRecorder`][kudio.core.stream.StreamRecorder] 在 PortAudio 的執行緒上擷取，保留每個樣本；
GUI 裡由按鈕呼叫 [`start()`][kudio.core.stream.StreamRecorder.start] 和
[`stop()`][kudio.core.stream.StreamRecorder.stop]。`sr=` 只是請求：裝置不接受時，依序試 16000、
22050、44100 Hz，再試裝置的預設值，選定的結果記在 INFO。`rec.sr` 和 `rec.channels` 才是錄音
實際的格式。打不開的輸入會丟出 `DeviceError`。

| 任何執行緒都能安全呼叫 | 傳回 |
|---|---|
| [`level()`][kudio.core.stream.StreamRecorder.level] | 最新一塊的峰值，0 到 1 |
| [`level_db()`][kudio.core.stream.StreamRecorder.level_db] | 同上，單位 dBFS，數位靜音是 `-inf` |
| [`elapsed()`][kudio.core.stream.StreamRecorder.elapsed] | 已錄的秒數 |
| [`tail(seconds)`][kudio.core.stream.StreamRecorder.tail] | 最近 *seconds* 秒，給畫面用 |
| [`drain()`][kudio.core.stream.StreamRecorder.drain] | 上次 `drain()` 之後的所有樣本，每個樣本只給一次 |
| [`take()`][kudio.core.stream.StreamRecorder.take] | 到目前為止的全部，不停止錄音 |

一塊是 `frame_size`（1024）個樣本。`stop()` 之後再 `start()` 會接著錄同一段，像暫停鍵一樣；
裝置回來時取樣率變了會丟出 `DeviceError`；[`reset()`][kudio.core.stream.StreamRecorder.reset]
丟掉這段錄音。錄到 `max_seconds`（3600，`None` 表示不設上限）就不再變長，
[`capped`][kudio.core.stream.StreamRecorder.capped] 變成 true，音量仍會更新。`monitor=True`
會把輸入同時送到預設輸出。會保留狀態的東西，頻譜圖、降噪器或寫檔，都要從 `drain()` 餵；
`tail()` 傳回的是互相重疊的視窗。

## 選擇裝置 {#choosing-a-device}

每個裝置只有一個編號，也就是 sounddevice 的編號：[`list_devices()`][kudio.util.check.list_devices]、
[`CheckDevice().system_devices()`][kudio.util.check.CheckDevice.system_devices] 和
`kudio devices` 都印出它和所屬的 host API，`record`、`Recorder`、`StreamRecorder` 和
`LocalStreamReader` 都以 `device` 收它。沒給時，裝置由
[`CheckDevice.default_input`][kudio.util.check.CheckDevice.default_input] 決定：系統的預設輸入，
它啟動不了就丟出 `DeviceError`；沒有預設輸入時（例如麥克風全在 WDM-KS 上的 Windows），用最後
一個打得開也啟動得了的輸入。在乎由哪個輸入錄音時，請傳 `device`。
[`check_available_rate(device)`][kudio.util.check.CheckDevice.check_available_rate] 傳回十二個
標準取樣率裡打得開的那些。

## 即時頻譜圖：`SpectrogramStream` {#a-live-spectrogram-spectrogramstream}

```python
import numpy as np
import kudio

sr = 16000
y = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(2 * sr) / sr)).astype(np.float32)
view = kudio.SpectrogramStream(sr, seconds=3.0)
print(view)
for start in range(0, len(y), 700):        # whatever block size the device uses
    new = view.push(y[start:start + 700])  # the columns this block completed
print(view.columns().shape, view.filled, view.frames_seen)
print(view.latency_seconds, view.column_seconds, round(float(view.latest().max()), 2))
```

```text
SpectrogramStream(sr=16000, n_fft=512, hop_length=128, 257 bins, 3.00s window)
(375, 257) 247 247
0.032 0.008 -6.02
```

[`SpectrogramStream`][kudio.core.spectrogram.SpectrogramStream] 在每一欄的樣本到齊時只算一次，
並用環狀緩衝區保留最近 `seconds` 秒。[`push()`][kudio.core.spectrogram.SpectrogramStream.push]
收任意長度的單聲道區塊，傳回它完成的欄，單位 dB，舊的在前；區塊大小不會改變任何一個值。
[`columns()`][kudio.core.spectrogram.SpectrogramStream.columns] 是整個視窗，從第一次呼叫起就是
完整寬度，還沒收到的部分是 [`floor_db`][kudio.core.spectrogram.SpectrogramStream.floor_db]；
[`latest()`][kudio.core.spectrogram.SpectrogramStream.latest] 是最新的一欄。第一欄要等一個音框，
`n_fft / sr`，之後每隔 `hop_length`（`n_fft // 4`）一欄。

![同一個串流的三張 SpectrogramStream.columns() 快照，分別在 1、2.5 和 5 秒：視窗一直是完整寬度，還沒聽到的部分在底部，舊的音節從左邊捲出去](../images/live-audio-filling-light.webp#only-light)
![同一個串流的三張 SpectrogramStream.columns() 快照，分別在 1、2.5 和 5 秒：視窗一直是完整寬度，還沒聽到的部分在底部，舊的音節從左邊捲出去](../images/live-audio-filling-dark.webp#only-dark)

- 線性軸上的位準是 dBFS：滿刻度的正弦波是 0 dB，比 `ref` 低 `top_db`（80）以上的都壓到
  `floor_db`。畫的時候用固定的色彩範圍，`vmin=view.floor_db, vmax=0`。
- `n_mels=64` 讓列落在梅爾刻度上，比較看得出語音，但不是 dBFS：滿刻度的 1 kHz 正弦波大約是
  -18 dB，所以 `vmax` 設在 -15 左右。
- 第 *k* 欄涵蓋樣本 `[k*hop, k*hop + n_fft)`，不像離線 `STFT` 那樣置中，所以比同編號的離線欄
  晚半個音框；[`to_stft()`][kudio.core.spectrogram.SpectrogramStream.to_stft] 給出對應的幾何
  設定。
- 它沒有鎖：在同一個執行緒裡 push 和畫，跨執行緒的部分交給 `StreamRecorder.drain()`。2-D 的
  區塊會被 `FeatureError` 拒絕。

有 matplotlib（`kudio[viz]`）和麥克風時，即時畫面就是下面這段；要看降噪後的畫面，改成 push
[`StreamEnhancer`][kudio.enhance.streaming.StreamEnhancer] 的 `enhancer.process(rec.drain())`
（[降噪](enhancement.md)）：

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rec = kudio.StreamRecorder(sr=16000).start()
view = kudio.SpectrogramStream(rec.sr, seconds=3.0)
fig, ax = plt.subplots()
image = ax.imshow(view.columns().T, origin="lower", aspect="auto",
                  vmin=view.floor_db, vmax=0.0,
                  extent=(-view.window_seconds, 0.0, 0.0, rec.sr / 2))


def update(_):
    view.push(rec.drain())
    image.set_data(view.columns().T)
    return [image]


anim = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
plt.show()                                 # returns when the window is closed
y = rec.stop()
```

## 播放音訊 {#playing-audio}

```python
tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)    # float64
print(kudio.play_audio(tone, 16000))
```

```text
unsupported dtype: float64
```

[`play_audio`][kudio.core.stream.play_audio] 收陣列和取樣率、路徑，或開啟中的 `wave.Wave_read`；
大部分的失敗不丟例外，而是傳回一段說明，所以要檢查傳回值；`None` 表示播放了。陣列必須是 int16
或 float32。陣列在背景播放，腳本下一行就結束的話聲音會被切掉；`sounddevice.wait()` 會等到播完。
路徑照 `file_load` 的方式讀。沒有 sounddevice 時丟出 `DependencyError`。

## 底層元件 {#the-lower-level-pieces}

[`Recorder`][kudio.core.stream.Recorder] 是阻塞式、固定長度、傳回 int16 的錄音器，取樣率不會
退而求其次：取樣率被拒絕或裝置無法啟動時，和 `record()` 一樣丟出寫明裝置編號和取樣率的
`DeviceError`；裝置不再送資料時，在錄音該結束的十秒後也丟出 `DeviceError`。

```python
with kudio.Recorder(record_times=3, rate=16000) as r:   # the system's default input
    y = r.start_record()                       # int16, (n,) for mono
    r.save("take.wav")                         # 16-bit WAV; True or False
```

[`LocalStreamReader`][kudio.core.stream.LocalStreamReader] 開啟一個輸入，把每一塊（一維、聲道
交錯，int16，`wav_format=kudio.PA_FLOAT32` 時是 float32）交給 `AudioBuffer`、`on_frame` 回呼，
或兩者。打不開的裝置會讓它維持未初始化，`get_status()` 會說明；`get_setup()` 傳回實際開啟的
設定。`on_frame` 在 PortAudio 的執行緒上執行，所以存下資料就返回；裡面丟出的例外會被記錄，串流
繼續。

[`AudioBuffer`][kudio.core.buffer.AudioBuffer] 是容量 `size`（2）塊的有界 FIFO，滿了就丟掉最舊
的，適合畫面用。[`get()`][kudio.core.buffer.AudioBuffer.get] 等一塊，超過 `timeout` 秒丟出
`TimeoutError`；[`get_fixed_size_data(n)`][kudio.core.buffer.AudioBuffer.get_fixed_size_data]
等接下來的 `n` 塊，拒絕 `n = 0`。
[`FetchBuffer(buf).get_data(n, timeout=...)`][kudio.core.buffer.FetchBuffer.get_data] 在自己的
執行緒上做同樣的等待。

[`RemoteStreamReader`][kudio.core.stream.RemoteStreamReader] 是 TCP 用戶端，給以原始 PCM
串流的裝置用，格式是 `"16bit"` 或 `"32bit"`（浮點）：

```python
buf = kudio.AudioBuffer(size=16)
remote = kudio.RemoteStreamReader(buf)
if remote.connect_remote_device("192.168.1.50", 5000):
    remote.initialize_params(frame_size=2048, wav_format="16bit")
    remote.start()
    chunk = buf.get(timeout=5.0)               # 2048 int16 samples
    remote.terminate()                         # sends "stop", closes the socket
```

連線一開啟，伺服器就送樣本，收到文字 `stop` 時應該停止。其他指令請在 `start()` 之前用
[`sock_send()`][kudio.core.stream.RemoteStreamReader.sock_send] 送出。分成好幾段到達的一塊會被
拼回來，最多再讀 `break_count`（10）次。參數在 API 參考的[即時音訊](../reference/live-audio.md)。
