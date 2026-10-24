# 從 v2 遷移 {#migrating-from-v2}

kudio 3.0 是一個以整理為主的版本。它替特徵函式取了看名字就懂的名稱，把畫圖、
dataframe 和即時音訊移出預設安裝，並移除了 v2 還留著的相容性模組。大部分 v2
程式碼在 3.x 上可以原樣執行，只會多出警告。需要注意的有四件事：新的環境，因為
2.0 只能裝在 Python 3.8 上；匯入已移除的模組，以及已經不在 `kudio` 命名空間裡的
名稱；有些呼叫需要的套件，核心安裝已經不再帶入；還有 PESQ，它現在來自另一個
套件。

之後的 3.x 版本讓幾個呼叫在同樣的寫法下得到不同的結果，其中大多是修正。這些列在
最後面，因為現在才從 v2 升級的專案，會一次全部碰到。

## 升級 {#upgrading}

kudio 2.0 只發布了 CPython 3.8 的 wheel，而每個 3.x 版本都需要 Python 3.9 以上，
所以裝著 2.0 的環境裝不了 3.x。在那個環境裡執行 `pip install -U kudio`，會停在
2.0，而且什麼都不會說。請用 3.9 以上建立新的環境，裝到那裡面：

```bash
python -m venv .venv                # this python must be 3.9 or newer
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install kudio                   # plus the extras you use, see below
python -c "import kudio; print(kudio.__version__)"
```

你自己的程式碼也會跟著搬過去。2.0 把 numpy 鎖在 1.20 或更舊、scipy 鎖在 1.5.4
或更舊、librosa 鎖在 0.7.2 或更舊；目前的版本需要 numpy 1.24 以上且低於 2.6、
scipy 1.10 以上、librosa 0.10 以上，並透過 soundfile 讀檔。

## 找出舊名稱 {#finding-the-old-names}

每個舊名稱都還能呼叫，並會發出一個 `DeprecationWarning`，寫出該用的新名稱：

```text
DeprecationWarning: kudio.w2s() is deprecated and will be removed in a future release; use kudio.waveform_to_spectrogram() instead.
```

只有當呼叫寫在你執行的那個腳本裡時，Python 才會印出 `DeprecationWarning`。同樣的
呼叫如果在你自己套件的某個模組裡，預設是不會顯示的，而大部分 v2 呼叫正好都在
那裡。pytest 會在警告摘要裡列出它們，所以跑一次測試是把它們找出來最快的方法。
要把每一個都變成錯誤，讓碰到它的測試失敗：

```bash
python -m pytest -W "error:kudio.:DeprecationWarning"
python -W "error:kudio.:DeprecationWarning" your_script.py
```

這個過濾條件只比對訊息以 `kudio.` 開頭的警告，所以其他函式庫的棄用警告仍然只是
警告。

## 改名的函式 {#renamed-functions}

| v2 名稱 | 3.x 名稱 |
|---|---|
| `w2s`、`wavform2spec` | [`waveform_to_spectrogram`][kudio.core.feature.waveform_to_spectrogram] |
| `spec2wavform` | [`spectrogram_to_waveform`][kudio.core.feature.spectrogram_to_waveform] |
| `f2s`、`wav2spec` | [`file_to_spectrogram`][kudio.core.feature.file_to_spectrogram] |
| `s2w`、`spec2wav` | [`save_spectrogram_as_wave`][kudio.core.feature.save_spectrogram_as_wave] |
| `w2mfcc`、`wav2mfcc` | [`mfcc`][kudio.core.feature.mfcc] |
| `concat_mfcc`、`concat_mfcc_` | [`mfcc_from_files`][kudio.core.feature.mfcc_from_files] |
| `concat_logspec`、`concat_logspec_`、`contextual_LogSpectrogram` | [`logspec_from_files`][kudio.core.feature.logspec_from_files] |
| `wav2mel` | [`melspectrogram`][kudio.core.feature.melspectrogram] |

每個舊名稱現在都是新函式外面包一層警告，所以它接受新函式的參數，回傳新函式的
結果。這十五個名稱都還能從 `kudio` 和 `kudio.core.feature` 匯入，之後的版本會
移除。有兩個地方還是可能讓你踩到。`wav2mel(file_name=...)` 會失敗，因為
`melspectrogram` 把那個參數叫作 `source`：請用位置參數傳路徑，或寫成
`source=`。另外，STFT 相關函式在 3.2 改了 `win_length` 的預設值，這一頁最後一部分
會說明。

主控台的上色輔助函式，從頂層呼叫時也會警告：`kudio.red`、`kudio.warn`、
`kudio.error`、`kudio.info`、`kudio.h1` 和 `kudio.util.colors` 裡的其他函式，
還有拼錯的 `upercase`。它們仍然留在 `kudio.util` 裡，在那裡呼叫
`kudio.util.red(...)` 不會有警告，而 `upercase` 的正確拼法是 `uppercase`。

## 移除的模組 {#removed-modules}

3.0 移除了四個相容性模組，現在匯入它們會以 `ModuleNotFoundError` 失敗：

| 已移除 | 請改用 |
|---|---|
| `kudio._config` | [`KudioConfig`][kudio.config.KudioConfig]，也可以寫成 `kudio.KC` |
| `kudio._enh` | [`trad_enhance`][kudio.enhance.traditional.trad_enhance]、[`wavelet_low_pass_filter`][kudio.enhance.traditional.wavelet_low_pass_filter] |
| `kudio.version` | `kudio.__version__`、`kudio.__author__`；沒有 `__date__` |
| `kudio.util.others` | [`find_duplicate`][kudio.util.tools.find_duplicate] |

這些替代品全都可以直接從 `kudio` 匯入。

[`KudioConfig.default_config`][kudio.config.KudioConfig.default_config] 只放音訊
設定，分在 `waveform`、`device` 和 `f_mfe` 底下；訓練相關的鍵已經拿掉了，
`load_config` 的 `type=` 參數也是。原本從 kudio 預設值讀取模型或訓練設定的專案，
應該把這些設定放在自己的 YAML 檔裡，再用
[`KudioConfig.load_config`][kudio.config.KudioConfig.load_config] 載入。請先確認
檔案存在。路徑找不到時，`load_config` 會回傳音訊的預設值並發出 `UserWarning`，
而設定裡沒有的鍵讀出來會是空的設定，不是錯誤，所以路徑打錯之後，`cfg.train.lr`
會是 `{}`。要檢查某個區段在不在，請用 `"train" in cfg`，或呼叫 `cfg.freeze()`，
讓缺少的鍵引發 `KeyError`。

## 不再一起安裝的套件 {#packages-that-are-no-longer-installed}

kudio 2.0 安裝時會連同 matplotlib、pandas 和 PyAudio 一起裝。3.x 一個都不裝，
`import kudio` 也不會匯入其中任何一個。透過 kudio 使用它們的程式碼，需要對應的
選用套件組：

```bash
pip install "kudio[audio,viz,data]"
```

| 需要的功能 | 選用套件組 | 沒有安裝時 |
|---|---|---|
| `Recorder`、`LocalStreamReader`、`RemoteStreamReader`（從 `initialize_params()` 開始）、`play_audio`、`CheckDevice`（[錄音與即時檢視](guide/live-audio.md)） | `[audio]` | `DependencyError`，訊息寫出 `kudio[audio]`；它同時也是 `ImportError` |
| [`kudio.util.visual`][kudio.util.visual]：`WaveVisualizer`、`history_plot`、`Common`、各個 `plot_*` 輔助函式 | `[viz]` | 匯入這個模組時引發 `DependencyError` |
| [`AudioEvaluate.get_df`][kudio.core.evaluator.AudioEvaluate.get_df] 和 [`per_file`][kudio.core.evaluator.AudioEvaluate.per_file] | `[data]` | 呼叫時引發 `DependencyError` |
| [`data2xlsx`][kudio.util.conv.data2xlsx] | `[data]` | 呼叫時引發 `DependencyError` |

`DependencyError` 會寫出需要的選用套件組：

```text
kudio.exceptions.DependencyError: 'matplotlib' is required for this feature. Install it with: pip install kudio[viz]
```

有四個名稱光裝選用套件組還不夠。2.0 匯入 `kudio` 時會一起匯入 `WaveVisualizer`、
`history_plot`、`Common` 和 `data2xlsx`，3.x 則沒有把任何一個留在 `kudio`
命名空間裡，所以即使裝了 `[viz]`，`kudio.WaveVisualizer(...)` 還是會以
`AttributeError` 失敗。請從它們各自的模組匯入：

```python
from kudio.util.conv import data2xlsx
from kudio.util.visual import Common, WaveVisualizer, history_plot
```

`[audio]` 也會安裝 sounddevice，比較新的 [`record`][kudio.core.stream.record] 和
[`list_devices`][kudio.util.check.list_devices] 用的就是它；`[data]` 則帶入寫
`.xlsx` 檔需要的 openpyxl。`AudioEvaluate.eval` 不需要 pandas 也能執行，評估器的
[`rows`][kudio.core.evaluator.AudioEvaluate.rows] 屬性為每個增強後的檔案存一個
普通的 dict，給不想用 dataframe 的程式碼使用。含雜訊基準的逐檔分數只能從
`per_file()` 取得；沒有 pandas 時，你拿到的是它們的平均值，像是 `base_stoi`
等等。

## PESQ

PESQ 用的是 PyPI 上的 `pesq` 套件，由它自己的選用套件組安裝：

```bash
pip install "kudio[eval,pesq]"           # [pesq] compiles: it needs a C compiler
```

`[pesq]` 不包含在 `[eval]` 或 `[all]` 裡：它要從原始碼建置，會讓這兩組在沒有
編譯器的電腦上裝不起來。3.x 不使用 pysepm，也從來不會在執行期間安裝任何東西。

沒有 `[pesq]` 時什麼都不會失敗。[`AudioEvaluate`][kudio.core.evaluator.AudioEvaluate]
會在 `get_df()` 和 `per_file()` 裡省略 PESQ 欄，
[`eval_metrics`][kudio.core.evaluator.eval_metrics] 則在那個位置回傳 `None`。
唯一的提示是一行記錄，在設定 logging 之前都看不到（[入門](getting-started.md)
有示範怎麼設定）。可以用
[`check_metrics_install`][kudio.core.evaluator.check_metrics_install] 檢查，或用
`kudio.score(..., ["pesq"])` 指名要求 PESQ，這樣就會引發例外。

PESQ 預設是 MOS-LQO，16 kHz 音訊用寬頻。請用 3.x 把 v2 的輸出重新評分一次，
不要把新數字放在舊表格旁邊比，因為兩者不一定在同一個尺度上。大多數語音增強論文
引用的原始窄頻 P.862 分數，只差兩個參數：

```python
ev = kudio.AudioEvaluate(clean_files, noisy_files,
                         pesq_mode="nb", pesq_scale="raw")
```

[`pesq`][kudio.core.evaluator.pesq] 用 `mode=` 和 `scale=` 接受同樣的兩個選擇，
[`score`][kudio.core.evaluator.score] 和 `eval_metrics` 則是 `pesq_mode=` 和
`pesq_scale=`。ITU 的 `pesq.exe` 執行檔印出的也是這個原始分數，
[`pesq_score_exe`][kudio.core.evaluator.AudioEvaluate.pesq_score_exe] 仍然可以
執行它，前提是在 Windows 上並且裝了 `[data]`。該回報哪一種尺度，請看
[品質評估](guide/metrics.md)。

## 同樣的呼叫，結果不同 {#results-that-change-for-the-same-call}

這些不是改名：呼叫的寫法一樣，輸出卻不同。標題有標版本的，
[更新紀錄](changelog.md)裡那個版本的段落有說明這項變更。

### `win_length` 跟著 `n_fft` 變（3.2） {#win_length-follows-n_fft-32}

以前不管 `n_fft` 是多少，`win_length` 的預設都是 512。現在它跟著 `n_fft`，
和 librosa 一樣，適用於 `waveform_to_spectrogram`、`spectrogram_to_waveform`、
`file_to_spectrogram`、`save_spectrogram_as_wave` 和 `logspec_from_files`。
`n_fft` 維持預設 512 的呼叫不受影響。`n_fft=1024` 現在用 1024 個樣本長的窗分析，
以前則是用 512 個樣本的窗再補零；所以要重現模型訓練時用的特徵，請傳入舊的值：

```python
spec = kudio.waveform_to_spectrogram(y, n_fft=1024, win_length=512)
```

同一項變更也讓小於 512 的 `n_fft` 終於能用；以前 `n_fft=256` 會在 librosa 內部
引發例外。

### `Synthesizer` 保留乾淨檔案的取樣率（3.1） {#synthesizer-keeps-the-clean-files-rate-31}

[`Synthesizer`][kudio.core.synth.Synthesizer] 以前會把每份混音重新取樣到
16 kHz，所以 8 kHz 的語料庫會輸出成 16 kHz 的檔案。現在每個輸出檔都採用它那個
乾淨檔案的取樣率，雜訊則重新取樣來配合。原本就是 16 kHz 的語料庫沒有任何改變。
要繼續得到 16 kHz：

```python
syx.syn(mode="inc", target_sr=16000)
```

### `save_spectrogram_as_wave` 保留含雜訊檔案的取樣率 {#save_spectrogram_as_wave-keeps-the-noisy-files-rate}

它以含雜訊檔案本身的取樣率讀取（v2 則是以 16 kHz 載入），並以同樣的取樣率寫出
結果。請用檔案原本取樣率的音訊計算頻譜圖，或先把檔案重新取樣。

### 超過滿刻度時削峰，不會繞回（3.7） {#past-full-scale-clips-instead-of-wrapping-around-37}

以前比滿刻度還大聲的混音，在 16-bit 檔案裡會繞回：1.2 會變成 -0.8。現在
`Synthesizer`、[`save_wave`][kudio.core.io.save_wave] 和
`save_spectrogram_as_wave` 會在滿刻度削峰，並把 16-bit 樣本捨入到最接近的
量化階，`syn()` 也會發出警告，寫出被削峰的檔案數。
`Synthesizer(subtype="FLOAT")` 會保留超過 1.0 的所有內容。

### `file_load(..., mono=False)` 是聲道在後（3.0、3.3） {#file_load-monofalse-is-channels-last-30-33}

在 v2 裡，用 `mono=False` 載入的多聲道檔案會以 `(channels, frames)` 回傳，和
librosa 的回傳方式一樣。3.x 不論有沒有 `sr=`，都回傳 `(frames, channels)`：3.0
改了原生取樣率的那條路徑，3.3 改了重新取樣的那條。以前用 `y[0]` 取第一個聲道的
程式碼，現在要寫成 `y[:, 0]`；`y[0]` 是第一個 frame，而且不會有任何錯誤提醒你。
預設的單聲道載入不受影響。細節在[音訊檔案](guide/audio-files.md)。

### 讀不了的檔案會引發例外（3.0） {#a-file-that-cannot-be-read-raises-30}

[`file_load`][kudio.core.io.file_load] 會引發
[`AudioIOError`][kudio.exceptions.AudioIOError]，v2 則是印一則訊息並回傳 `None`。
原本檢查結果是不是 `None` 的程式碼，應該改成攔截這個例外。

### `reg` 模式搭配雜訊和 SNR 的方式不同（3.7） {#reg-mode-pairs-noise-and-snr-differently-37}

在預設模式下，[`Synthesizer.syn`][kudio.core.synth.Synthesizer.syn] 以前是讓
雜訊清單和 SNR 清單並排、各自輪流。兩者的數量有公因數時，有些配對永遠不會出現：
14 種雜訊對 21 個 SNR，每種雜訊只會配到 3 個 SNR。現在每輪完一圈這樣的配對，SNR
就多往後錯開一格，所以任何連續「雜訊數 × SNR 數」個輸出裡，每種配對都剛好出現
一次。兩者數量沒有公因數時，沒有任何改變。

### `is_silence=True` 會加入一部分純雜訊檔案（3.7） {#is_silencetrue-adds-a-share-of-noise-only-files-37}

以前這個旗標會傳給每一個檔案，所以每個輸出都被寫成只有雜訊，而一律會設定這個
旗標的 `syn_extra_mode`，從來沒產生過含語音的混音。現在只有 `p_silence` 那個
比例的輸出，會另外再寫一份純雜訊，混音則保留語音。3.7 之前用這個旗標建立的
資料集需要重建。

## 從 3.0 到 3.6 之間的版本升級 {#coming-from-30-to-36}

還有兩項變更只跟為 3.0 到 3.6 寫的程式碼有關；2.0 既沒有這個關鍵字參數，也沒有
種子。

- `Synthesizer.syn(desired_sample=16000)` 現在是
  `Synthesizer.syn(target_sr=16000)`。`desired_sample` 看起來像樣本數，意思卻是
  取樣率。它還能用，但會發出警告，訊息以 `Synthesizer.syn` 開頭，所以請在上面的
  過濾條件加上 `-W "error:Synthesizer.syn:DeprecationWarning"` 來抓它。
- 3.7 之前用某個種子建立的語料庫，沒辦法再用那個種子重建。現在每個輸出檔的雜訊
  截取位置、房間和封包遺失，都由種子加上它自己的路徑決定，所以指定了種子的執行，
  不論有沒有用行程池，都會得到相同的檔案。執行完把
  [`Synthesizer.seed`][kudio.core.synth.Synthesizer.seed] 留下來，就能再建出同一份
  語料庫；[建立含雜訊資料集](guide/synthesis.md)有示範。
