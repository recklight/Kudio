# 指標的驗證方式 {#how-the-metrics-are-checked}

下表的每個指標，都拿去和算出文獻數字的那份實作比對過；沒有這種實作的，就用第二種
方式把定義重新算一遍來比對。`tonal_lines` 和 `audio_report` 的頻寬上限、削波計數、
樣本格點兩者都沒有，就用答案已知的訊號檢查。表下各節說明定義出自哪裡，最後幾節說明
怎麼執行 `tools/validation/` 裡的腳本。沒有註明「相對」的差異都是絕對差異。

測試裡存著參考實作在測試自己的配對上算出的數字，所以不用安裝參考實作，`pytest` 也會
拿這些數字檢查 kudio；SRMRToolbox 的比值、PESQ 的一致性測試和 pySIIB 的範例圖只留給
腳本，因為它們的錄音沒有收進版本庫。

| 指標 | 比對對象 | 最大差異 |
|---|---|---|
| `llr`、`cepstral_distance`、`wss`、`fw_segmental_snr` | pysepm_evo 0.1.1 | 相對 1.1e-11 |
| `itakura_saito` | 重寫的 comp_is.m；兩個 LPC 頻譜的 Itakura-Saito 散度 | 相對 4.1e-11 |
| `composite` | 以 pysepm_evo 0.1.1 的各部分和 pesq 0.0.4 代入的迴歸式 | 相對 3.3e-11 |
| `lsd` | URGENT 2025 和 `ssr_eval` 的定義，在 librosa 的 STFT 上重寫 | 相對 1.1e-10 |
| `mcd` | 在 pysptk 1.0.1 和 fastdtw 0.3.4 上的 URGENT 2025 MCD | 7.1e-15 dB |
| `srmr` | SRMRToolbox 為它的測試錄音存的四個比值 | 相對 4.5e-13 |
| `wada_snr` | 用另一種方式算出的模型 | 讀取 SNR 的那條曲線上 2.3e-11 |
| `pesq` | ITU-T P.862 Annex A 的一致性測試 2(b) | 5.0e-4，也就是表格的捨入 |
| `dnsmos`、`dnsmos_p808` | DNS-Challenge 的 `dnsmos_local.py`，在它釘的套件版本下執行；torchmetrics 1.9.0 | 6.8e-7；和 torchmetrics 相比為 0 |
| `ncm` | pysepm_evo 0.1.1；ANSI S3.5-1997 表格的兩份副本 | 16 kHz 為 2.2e-6，8 kHz 為 8.3e-9；表格數值完全相同 |
| `csii` | pysepm_evo 0.1.1；ANSI S3.5-1997 表格的兩份副本 | 4.4e-16；表格數值完全相同 |
| `siib`、`siib_gauss` | 用第二種方式重算的兩個指標；commit 226c2f3 的 pySIIB；pySIIB 範例圖上的點 | 16 kHz 時，SIIB^Gauss 和兩者的相對差都在 1e-12 以下，SIIB 相對 1.4e-5，也就是打破平手的雜訊；其他取樣率是 pySIIB 的重新取樣，差幾 bit/s；15 個點全在圖的讀數範圍內，SIIB 要用 pySIIB 的濾波器轉到 16 kHz |
| `wer`、`cer` | 列舉所有對齊；完整的編輯距離表 | 0 |
| `eer`、`min_dcf`、`det_curve` | 以有理數精確計算的掃描；scikit-learn；VoxSRC 和 NIST 的評分程式 | EER 為 0；min DCF 為 2.2e-16 |
| `cllr`、`min_cllr`、`act_dcf`、`c_primary` | 40 位數精度；PYLLR 的規則；scikit-learn 的 isotonic regression；有理數精確計算 | 3.3e-16 bit；成本 2.8e-14 |
| F0 指標 | mir_eval 0.8.2；逐音框計數 | 計數為 0；比率為 1.1e-16 |
| `der`、`jer`、`detection_error_rate` | pyannote.metrics 4.2.0；dscore 的 JER；1 ms 音框計數 | 8.9e-16 |
| `pit_si_sdr`、`si_sdr_improvement`、`bss_eval`、`sdr_improvement` | torchmetrics 1.9.0；所有排列；mir_eval 0.8.2 | SI-SDR 為 8.9e-11 dB；BSS Eval 為 0 |
| `bootstrap_ci`、`paired_ci`、`rate_ci`、`paired_rate_ci`、`permutation_test`、`holm` | scipy 的 `bootstrap` 和 `permutation_test`（1.13.1 和 1.18.1），同一批重抽樣和符號組合；有理數精確值；重寫的定義 | 從 scipy 的重抽樣讀出的百分位數兩端和精確 p 值為 0；BCa 為寬度的 2.4e-14 |
| `audio_report` 的 `band_limited`、`source_rates`、`mirrored` | 沒有可比對的實作；來歷已知的 720 個升取樣檔案、10 個編解碼器截頻和 470 段錄音 | 0 和 -20 dB 時 480 個標出 468 個，沒標出的 12 個是取樣率比不是整數倍的線性內插；沒有指名錯的；470 段錄音沒有一段被標出 |
| `tonal_lines` | 沒有可比對的實作；720 個正弦波、90 段雜訊、40 組梳狀 | 正弦波上 0.162 Hz、0.52 dB；雜訊裡沒有線；40 組梳狀都分對 |
| `audio_report` 的 `clipped_samples`、`full_scale_samples` | 沒有可比對的實作；1000 個平滑波峰、409 個削掉的正弦波、一段過載的人聲 | 沒有平滑波峰被算到；推到 1.01 倍以上的正弦波，平台全部完整算到 |
| `audio_report` 的 `sample_step`、`grid_bits` | 沒有可比對的實作；7 種已知格點和 4 段沒有格點的 | 0 |

## Hu 與 Loizou 的指標和綜合指標 {#hu-and-loizous-measures-and-the-composite}

定義出處
:   Hu 和 Loizou 2008：[^hl08] LLR 式 4、IS 式 5、CEP 式 6-7（式 7 是 Kitawaki 等人的 LPC 倒頻譜距離，即他們的式 1[^kit88]）、
    fwSegSNR 式 8-9、WSS 式 10（依 Klatt[^klatt82]）；綜合指標的迴歸式在 Hu 和 Loizou
    2006[^hl06] 第 4.1 節。視窗、eps、音框數和捨入照 Loizou 的 MATLAB 檔案
    （comp_llr.m、comp_is.m、comp_cep.m、comp_wss.m、comp_fwseg.m、composite.m）。[^l13]

## 對數頻譜距離 {#log-spectral-distance}

定義出處
:   URGENT 2025 評分腳本裡的 `lsd_metric`[^urgent]
    （`evaluation_metrics/calculate_intrusive_se_metrics.py`，最後一次修改是 2024-11-19
    的 commit 93e16aa），以及 NVSR 的 `ssr_eval` 裡的 `AudioMetrics.lsd`[^nvsr]
    （`ssr_eval/metrics.py`，最後一次修改是 2022-03-28 的 commit 907395c），兩者都在
    2026-11-13 讀取。

## 梅爾倒頻譜失真 {#mel-cepstral-distortion}

定義出處
:   URGENT 2025 評分腳本的 `mcd_utils.py` 和 `mcd_metric`[^urgent]（commit c726558
    和 93e16aa），移植自 ESPnet 的 MCD：SPTK 的 `mcep`，也就是 Fukada 等人的梅爾
    倒頻譜分析，[^ftki92] 以及 FastDTW。[^sc07]

## SRMR {#srmr}

定義出處
:   Falk、Zheng 和 Chan，[^srmr] 正規化版本則是 Santos、Senoussaoui 和 Falk，[^srmrnorm]
    照作者的 SRMRToolbox 在 `SRMR.m` 裡的算法（github.com/MuSAELab/SRMRToolbox，head
    為 2023-05-23 的 690bdd1）。

## WADA-SNR {#wada-snr}

定義出處
:   Kim 和 Stern，[^wada] 式 1、6、8 和 17，語音為形狀參數 0.4 的 gamma 分布，雜訊為
    高斯分布。

## DNSMOS {#dnsmos}

定義出處
:   DNS-Challenge 發布的三個模型：`DNSMOS/sig_bak_ovr.onnx`（P.835）[^dnsmos835]、
    `pDNSMOS/sig_bak_ovr.onnx`（個人化）和 `DNSMOS/model_v8.onnx`（P.808）[^dnsmos808]，
    取 commit 785ab12 的版本，照 `DNSMOS/dnsmos_local.py`（commit 6412bc5，2023-01-10）
    的方式執行，套件用它的 `requirements.txt` 釘的版本：numpy 1.22.4、librosa 0.8.1 和
    onnxruntime 1.13.1。

## PESQ {#pesq}

定義出處
:   ITU-T P.862，加上 P.862.1 對應到 MOS-LQO 的關係和 P.862.2 的寬頻延伸[^p862]。
    P.862 的 Annex A 在 2005 年由 Amendment 2 修訂，其中的 C 程式碼（2.0 版）是建議書的
    一部分，和文字敘述有衝突時以程式碼為準（A.1.1）。

## NCM 和 CSII {#ncm-and-csii}

定義出處
:   兩者都出自 Ma、Hu 和 Loizou，[^mhl09] CSII 另出自 Kates 和 Arehart，[^ka05] 照
    NCM.m 和 CSII.m 的算法（F. Chen 和 P. C. Loizou，2011 和 2012 年，收在 Loizou
    那本書附的程式碼裡[^l13]）。頻帶表取自 ANSI S3.5-1997，[^ansi] 並和 CRAN 套件
    SII 1.3.0[^siir] 及標準工作小組的 C 程式[^siic] 比對過。

## SIIB {#siib}

定義出處
:   Van Kuyk、Kleijn 和 Hendriks：SIIB 出自他們的短文，[^vkh18a] 式 2 和 8 到 12；
    SIIB^Gauss 出自他們對侵入式指標的評估，[^vkh18b] 第 IV-C 節。SIIB 每個通道的資訊量
    用 Kraskov、Stögbauer 和 Grassberger 的第二個估計器計算，[^ksg04] 也就是他們的
    式 9。論文沒講清楚或程式做法不同的地方，kudio 照 SIIB.m、SIIB_Gauss.m 和 MILCA 的
    MIxnyn；[API 參考][kudio.core.siib]列出每一處。

## WER 和 CER {#wer-and-cer}

定義出處
:   兩個詞序列或字元序列之間的 Levenshtein 距離，每個編輯的代價都是 1；WER 照
    Jurafsky 和 Martin 在這個距離上的定義。[^jm] 語料合併的方式和 Kaldi 的
    `compute-wer`、jiwer 相同。

## EER、最小 DCF 和 DET 曲線 {#eer-minimum-dcf-and-the-det-curve}

定義出處
:   DET 曲線的操作點；[^det] EER 以相鄰操作點之間的直線來讀，和 VoxSRC 2020-2023 的
    評分程式、NIST SRE 2016 評分腳本（4.1 版，WeSpeaker 收錄的版本）相同，或以凸包
    來讀，也就是 BOSARIS 的 ROCCH-EER[^bosaris]（第 2.6.2 節）；偵測成本和它的正規化
    照 NIST 評測計畫的定義，minDCF 是所有門檻中最低的正規化成本。[^nist]

## 校準：C_llr、最小 C_llr、實際 DCF 和 C_Primary {#calibration-c_llr-min-c_llr-the-actual-dcf-and-c_primary}

定義出處
:   C_llr 和用 pool-adjacent-violators 擬合算出的最小 C_llr 照 Brümmer 和
    du Preez；[^bd06] 門檻 log(beta) 上的實際成本 actDCF，以及它在 p_target 0.01 和 0.005
    上的平均 C_Primary，照 NIST SRE 2016 評測計畫的 3.1 節。[^nist]

## 語者分段標記：DER、JER 和偵測錯誤率 {#diarization-der-jer-and-the-detection-error-rate}

定義出處
:   DER 照 NIST RT-09 評測計畫 6.1 節的定義，[^rt09] 以及 md-eval.pl（第 22 版，dscore
    附的版本）的評分方式：在所有評估的時間上配對語者，之後才在每個參考邊界兩側去掉
    給定秒數的 collar，並視需要去掉重疊語音。JER 照 DIHARD 的評分程式 dscore 的
    算法；[^dihard] 偵測錯誤率照 pyannote.metrics 的 `DetectionErrorRate`。

## F0 誤差 {#f0-errors}

定義出處
:   GPE、VDE 和 FFE 照 Chu 和 Alwan[^ca09] 的式 1-5，FPE 照 Drugman 和 Alwan[^da11]
    的定義，四種情況出自 Rabiner 等人，[^r76] RMSE 則是以音分計的
    `1200 log2(est / ref)`。

## 分離：SI-SDRi、PIT 和 BSS Eval {#separation-si-sdri-pit-and-bss-eval}

定義出處
:   SI-SDR 照 `si_sdr` 的算法，兩個訊號先去掉平均值，和 Conv-TasNet 的 SI-SNR 相同
    （Luo 和 Mesgarani，式 15[^convtasnet]）；配對取所有音源平均 SI-SDR 最高的那一組，
    也就是語句層級 PIT 的準則；[^upit] SDR、SIR 和 SAR 照 BSS Eval；[^bss] 進步量是
    估計的分數減去混合訊號對同一個音源的分數。

## 信賴區間與符號翻轉檢定 {#confidence-intervals-and-the-sign-flip-test}

定義出處
:   百分位數區間和 BCa 區間，照 Efron 1987[^e87] 和 Efron 與 Tibshirani 1993[^et93]
    第 14 章，加速度由被重抽樣的單位做 jackknife 估計；整組有放回重抽樣的 cluster
    bootstrap，即 Field 與 Welsh 研究的作法；[^fw07] 錯誤率以總和之比重抽樣，和 Bisani
    與 Ney 處理 WER 的方式相同；[^bn04] 符號翻轉檢定，隨機抽取組合時 p 值用 Phipson 與
    Smyth 的 (b + 1)/(B + 1)；[^ps10] 以及 Holm 的逐步調整。[^h79]

## `audio_report` 的升取樣與頻寬上限 {#upsampling-and-band-limits-in-audio_report}

定義出處
:   kudio 自己的規則；沒有公開的檢定能判斷檔案是否被升取樣過。邊緣是這樣的頻率：
    Welch 頻譜（8192 個樣本的 Hann 窗分段、重疊一半，再以邊緣頻率的 2% 做中位數平滑）
    在邊緣的 0.70–0.88 倍處，比 1.05–1.25 倍處高出至少 20 dB，而且比從 1.05 倍到奈奎斯特
    頻率 0.97 倍的全部頻帶的中位數高出至少 40 dB。內容停在和邊緣下方頻帶相差不到 10 dB
    的最後一個頻率；內容停在某個常見取樣率奈奎斯特頻率的 0.95–1.06 倍，就指名那個取樣率。
    線性內插另外檢查：整數倍時，它在原取樣率的奈奎斯特頻率 fc 上方的鏡像，比它所鏡射的頻帶低
    `40 log10 |sin(pi (fc - x) / sr) / sin(pi (fc + x) / sr)|` dB，量到的差和這條曲線的
    中位數絕對偏差要在 0.3 dB 以內，中位數離曲線不超過 1 dB。

## 窄頻線：`tonal_lines` {#narrowband-lines-tonal_lines}

定義出處
:   kudio 自己的規則。線是 Welch 頻譜（Hann 窗分段、重疊一半，段長取不超過
    `sr / resolution_hz` 的最大 2 的次方，預設 5 Hz）上的局部峰，比周圍 101 格的中位數
    高出 `min_prominence_db`（20 dB）；頻率和位準由通過峰值三格（dB）的拋物線讀出。
    梳狀是落在同一個頻率一段不間斷整數倍上的三條以上的線，連續三個倍數沒有線就算中斷；
    每解釋一條線得一分，從 1 到最高成員之間每缺一個倍數扣半分。

## `audio_report` 的削波 {#clipping-in-audio_report}

定義出處
:   kudio 自己的規則。滿刻度 0.999 以上的樣本連成一段，一段的平台是離段內最高值 1e-4
    以內的樣本。平台長度至少 `clip_run`（預設 3）個樣本，而且佔它那一段的比例，至少是
    同樣高度的平滑波峰在頂端 1e-4 內所佔比例的兩倍，才算削波：頂端在 m 的平滑波峰是
    `sqrt(1e-4 / (m - 0.999))`，滿刻度時 0.32，所以那裡的門檻是 0.63。頂端離 0.999
    不到 1e-4 的段不判斷。`full_scale_samples` 計算每個在 0.999 以上的樣本。

## `audio_report` 的樣本格點 {#the-sample-grid-of-audio_report}

定義出處
:   kudio 自己的規則。每個有限值樣本以 24 位元的解析度讀取。全部都是整數，而且扣掉
    滿刻度的兩個值之後還有 16 個以上不同的值時，格距就是相鄰值之差的最大公因數；
    `sample_step` 是格距佔滿刻度的比例，`grid_bits` 是 `log2(2 / sample_step)`。少於
    15.95 位元算一項發現。

## SNR、SI-SDR、STOI 和 SDR {#snr-si-sdr-stoi-and-sdr}

`snr`、`si_sdr`、`segmental_snr` 和 `sdi` 各是幾行 numpy，定義在
[品質評估](guide/metrics.md#what-each-metric-sees)。`stoi` 和 `estoi` 是 pystoi，`sdr`
是照 mir_eval 做的 BSS Eval。kudio 把兩者裁到共同的長度再傳下去，所以它們的數字就是這些
套件的數字。

## kudio 和 librosa {#kudio-and-librosa}

重新取樣、短時傅立葉轉換與反轉換、梅爾濾波器、梅爾頻譜與 MFCC、靜音切割、pYIN 和時間
伸縮，都照著 librosa[^librosa] 的做法，由 kudio 自己在 numpy、scipy 和 soxr 上計算。
這些慣例，從音框怎麼補邊到梅爾濾波器怎麼縮放，都是 librosa 的，這要謝謝它的作者們；
librosa 以 ISC 授權釋出。演算法則引用它們最初發表的地方：反轉換是 Griffin 和 Lim 的
最小平方估計[^gl84]，梅爾刻度是 Slaney 的[^s98]，倒頻譜是 Davis 和 Mermelstein 的[^dm80]，
音高追蹤是建立在 YIN[^yin] 上的 pYIN[^pyin]，時間伸縮是相位聲碼器[^fg66][^ld99]。

`tools/validation/against_librosa.py` 用 `tests/test_against_librosa.py` 的案例、同樣的
種子，比對 kudio 和 librosa。差異 0 指逐位元相同：形狀、型別和每個位元都一樣，NaN 也在
同樣的位置。

| 對照 | 案例數 | 最大差異 |
|---|---|---|
| `librosa.resample` 和 `librosa.load(sr=...)`：13 組取樣率、8 種重新取樣器、1、2、6 聲道、12 種樣本型別、5 種記憶體排列、6 種檔案 | 1626 | 0 |
| `librosa.stft`：16 種音框與位移、8 種窗、float16、float32 和 float64，從一個樣本到 5001 個 | 1062 | 0 |
| `librosa.istft`：15 種音框與位移、3 種窗、1 到 40 個音框、complex64 和 complex128 | 1644 | 0 |
| 建在這兩者上的頻譜圖函式 | 128 | 0 |
| `librosa.filters.mel`：7 種取樣率、6 種音框、1 到 256 個頻帶、4 種最低和 5 種最高頻率 | 5880 | 0 |
| `librosa.mel_frequencies` | 30 | 0 |
| `librosa.feature.mfcc` 和 `librosa.feature.melspectrogram`：8 種片段、2 種精度、3 種取樣率 | 720 和 192 | 0 |
| `librosa.effects.trim` 和 `librosa.effects.split`：25 個片段、2 種精度、6 種門檻、7 種音框 | 各 2100 | 0 |
| `librosa.pyin` 的完整搜尋：528 種困難訊號，另外 1243 個涵蓋各種位移、音框、範圍、取樣率和長度 | 1771 | 0 |
| `librosa.effects.time_stretch`：8、16、44.1 kHz 的兩秒人聲、6 種倍率，結尾不算 | 18 | 對 0.11 是峰值的 0.47 %，對 1.0 是百萬分之 4.3 |

kudio 用整數算重新取樣後的長度。從 11025 Hz 這一族的取樣率轉到 12、24、48、96 或
192 kHz 時，librosa 用浮點數算的 `ceil(n * target_sr / orig_sr)` 會多出一個樣本，值是零；
上表比樣本時把它拿掉。

kudio 的 pYIN 每一個轉移都搜尋，和 librosa 0.11 一樣；librosa 1.0 除非傳入 `None`，會略過
低於 `transition_min_prob=1e-4` 的轉移，上表比的是它的完整搜尋。下表是 1.0 的預設有聲
判斷不同的音框數，依位移列出，材料是一分鐘有停頓的人聲（16 kHz、65-400 Hz）加白雜訊：

| 位移（ms） | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 45 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 雜訊底低 60 dB | 0/12001 | 0/6001 | 0/4001 | 0/3001 | 0/2401 | 0/2001 | 1/1501 | 5/1334 | 267/1201 | 63/1001 | 21/858 |
| 雜訊底低 40 dB | 0/12001 | 0/6001 | 0/4001 | 0/3001 | 0/2401 | 0/2001 | 1/1501 | 5/1334 | 267/1201 | 63/1001 | 20/858 |

時間伸縮本來就不打算和 librosa 一樣。kudio 的相位以 float64 累加，過了最後一個音框
之後保持振幅、相位繼續轉，所以在結尾，人聲和 librosa 0.11 最多差峰值的 0.8 %，和 1.0
差 1.1 %。48 kHz、float32 的純音拉長之後，每秒用頻率不固定的最佳正弦波去量，每一格寫的
是結尾時離原本的音差幾音分，以及結尾和最差的那一秒的 SINAD：

| 純音，拉長為 1.25 倍 | librosa 0.11.0 | librosa 1.0.1 | kudio |
|---|---|---|---|
| 187.3 Hz, 10 s | -0.032 音分; 98.1 / 58.5 dB | 0.000 音分; 99.0 / 93.8 dB | 0.000 音分; 99.1 / 93.8 dB |
| 3001.7 Hz, 10 s | +0.033 音分; 70.3 / 29.1 dB | 0.000 音分; 121.4 / 113.5 dB | 0.000 音分; 140.5 / 139.3 dB |
| 187.3 Hz, 300 s | +1.314 音分; 75.1 / 20.1 dB | 0.000 音分; 93.1 / 91.6 dB | 0.000 音分; 93.1 / 91.6 dB |
| 3001.7 Hz, 300 s | -1.514 音分; 64.4 / 3.9 dB | 0.000 音分; 97.7 / 97.6 dB | 0.000 音分; 117.7 / 117.7 dB |
| 20640 Hz, 300 s | +0.868 音分; 43.0 / -4.3 dB | 0.000 音分; 68.3 / 68.1 dB | 0.000 音分; 130.6 / 129.4 dB |

## kudio 和 PyWavelets {#kudio-and-pywavelets}

`wavelet_low_pass_filter` 用 kudio 自己的離散小波轉換（periodization 模式）拆開和組回片段，
106 個離散小波的濾波器是從 PyWavelets[^pywt] 照抄來的。PyWavelets 以 MIT 授權釋出，它的
授權聲明放在 `kudio/enhance/_wavelet_filters.py` 的開頭，這要謝謝它的作者們。每一個加總的
順序都和 PyWavelets 的 C 程式相同；coiflet 的 float32 濾波器是 PyWavelets 用它自己的表
以 float32 算出來的，不是 float64 的濾波器四捨五入，kudio 也用這些值，所以結果和 PyWavelets
逐位元相同。`tests/test_against_pywavelets.py` 拿 kudio 和它比；CI 上做比對的工作設了
`KUDIO_REQUIRE_PYWAVELETS=1`，PyWavelets 匯入不了就停下來，不讓測試跳過。

| 和 PyWavelets 比的 | 案例 | 差異 |
|---|---|---|
| 每個離散小波的四個濾波器，float64 和 float32 | 106 | 0 |
| `dwt` 和 `idwt` 各一層：1 到 20 個樣本的每一種長度，以及 31、101、1001，float32 和 float64 | 各 4876 | 0 |
| `wavedec`、`threshold(mode='soft')`、`waverec` 對 `wavelet_low_pass_filter`：1 到 4097 個樣本的 14 種長度、兩種門檻輪流、float32 和 float64 | 2862 | 0 |
| 同上，樣本是 float16、big-endian 的 float32 和 float64、int16、uint8 | 40 | 0 |
| `pywt.Wavelet`，以及用同樣的濾波器、取別的名字做出來的 | 20 | 0 |

## kudio 和 mir_eval {#kudio-and-mir_eval}

`score` 裡的 `sdr`、[`bss_eval`][kudio.core.separation.bss_eval] 和
[`sdr_improvement`][kudio.core.separation.sdr_improvement] 跑的 BSS Eval[^bss]，
一步一步、一個呼叫一個呼叫照著 mir_eval[^mireval] 0.8.2 的 `bss_eval_sources`，所以數字
和 mir_eval 逐位元相同，也不需要裝 mir_eval。mir_eval 以 MIT 授權釋出，它的授權聲明放在
`kudio/core/_bss.py` 的開頭，這要謝謝它的作者們。參考訊號讓正規方程式奇異時，例如兩個只有
兩個樣本、一個是另一個兩倍的參考訊號，mir_eval 0.8.2 會轉用最小平方解；但在 numpy 2.5 上，它用來攔這個錯誤的名稱
已經不在了，會變成 `AttributeError`，所以比對前先把那個名稱補回去。
`tests/test_against_mir_eval.py` 拿 kudio 和它比，CI 上做比對的工作設了
`KUDIO_REQUIRE_MIR_EVAL=1`。

| 和 mir_eval 0.8.2（numpy 2.5.5）比的 | 案例 | 差異 |
|---|---|---|
| `bss_eval_sources`，一個來源：1 到 3000 個樣本，float32 和 float64 | 12 | 0 |
| 兩個來源，有和沒有排列搜尋，5 到 1500 個樣本 | 12 | 0 |
| 三個來源，600 個樣本 | 4 | 0 |
| 讓正規方程式奇異的兩個參考訊號，用最小平方解 | 1 | 0 |
| 帶混音的 `bss_eval`、`score(['sdr'])`、`sdr_improvement` | 4 | 0 |

## kudio 和 pystoi {#kudio-and-pystoi}

`score` 裡的 `stoi` 和 `estoi`，一步一步照著 pystoi[^pystoi] 0.4.1 做：它重新取樣到
10 kHz 的方法、去掉靜音音框、三分之一八度頻帶，以及 30 個音框的片段。pystoi 以 MIT 授權
釋出，它的授權聲明放在 `kudio/core/_stoi.py` 的開頭，這要謝謝它的作者。有一步沒照做：
pystoi 的 ESTOI 在兩次正規化之前，各從 numpy 的全域亂數加一次機器精度大小的雜訊，這會
動到設了種子的腳本裡其他的亂數，也讓 ESTOI 的最後幾位數每次呼叫都不同。
`tests/test_against_pystoi.py` 拿 kudio 和雜訊換成零的 pystoi 比，也和未修改的 pystoi 比；
CI 上做比對的工作設了 `KUDIO_REQUIRE_PYSTOI=1`。

| 和 pystoi 0.4.1 比的 | 案例 | 差異 |
|---|---|---|
| STOI 和 ESTOI，雜訊換成零：8 到 48 kHz、0.5 到 4 秒、float32 和 float64 | 72 | 0 |
| 同上，邊界情形：剛好剩 30 個音框、停頓在峰值下 42 dB | 4 | 0 |
| STOI，未修改的 pystoi | 1 | 0 |
| ESTOI，未修改的 pystoi | 1 | 小於 1e-15 |
| `score(['stoi', 'estoi'])` | 2 | 0 |

## 執行這些腳本 {#running-the-scripts}

在 checkout 的最上層執行：

```bash
python tools/validation/quality.py
```

每個腳本都匯入它所在 checkout 裡的 kudio，印出每項比對的最大差異和容許誤差，有一項
超出時以 1 結束；缺少參考套件的列會略過並列出。pysepm 和 pySIIB 是 GPL-3.0，只在
kudio 旁邊執行，kudio 裡沒有它們的任何程式碼。

| 腳本 | 檢查 | kudio 以外需要的東西 | 時間 |
|---|---|---|---|
| `quality.py` | LLR、IS、CEP、WSS、fwSegSNR、composite | pysepm_evo、pesq；60 位數的 LLR 要 mpmath | 17 s；沒有它們時 4 s |
| `lsd.py` | `lsd` | librosa | 5 s |
| `melcepstrum.py` | `mcd` | pysptk、fastdtw | 2 分鐘 |
| `srmr.py [--toolbox-test DIR]` | `srmr` | SRMRpy；DIR 是 SRMRToolbox checkout 裡的 `test` 資料夾 | 有 DIR 時 53 s，沒有時 49 s；沒有 SRMRpy 時 1 s |
| `wada.py [--table FILE]` | `wada_snr` | 不需要；FILE 是 snreval 的 `Alpha0.400000.txt` | 14 s |
| `pesq_conformance.py --annex-a ZIP [--corrigendum-2 ZIP]` | `pesq` | pesq；ZIP 是 ITU 發佈的 P.862 Amendment 2 軟體檔和 Corrigendum 2 檔，照原樣使用 | 5 s |
| `dnsmos.py --reference-python PY [--models DIR]` | `dnsmos`、`dnsmos_p808` | onnxruntime；PY 是裝了 `dnsmos_local.py` 所釘套件的直譯器；DIR 是 DNS-Challenge checkout 裡的 `DNSMOS` 資料夾；有裝的話，torchmetrics 和 0.9 以後的 librosa | 2 分鐘；沒有 DIR 時 5 s |
| `intelligibility.py [--loizou-examples DIR]` | `ncm`、`csii` | pysepm_evo；DIR 放 NCM.m 的範例配對 | 約 4 分鐘 |
| `sii_tables.py --cran TARBALL --sii-c ZIP` | `ncm`、`csii` 裡 ANSI S3.5-1997 的數值 | xlrd；TARBALL 是 CRAN 套件 SII 1.3.0，ZIP 是 sii.to 的 C 程式開發套件，照原樣使用 | 1 s |
| `siib.py [--all]` | `siib`、`siib_gauss` 和第二種方式、和 pySIIB 比對，以及更陡的重新取樣濾波器讓它們變動多少；不含 pySIIB 的圖，那需要它的範例錄音 | 不需要；有裝 pySIIB 就會用，`pip install git+https://github.com/kamo-naoyuki/pySIIB.git`，會編譯 C 程式碼 | 2 分鐘；`--all` 時 11 分鐘 |
| `wer.py` | `wer`、`cer` | 不需要；裝了 nltk 會用 | 32 s，其中 23 s 花在一對 100,000 詞的配對上 |
| `verification.py` | `eer`、`min_dcf`、`det_curve` | scikit-learn | 9 s |
| `calibration.py` | `cllr`、`min_cllr`、`act_dcf`、`c_primary` | scikit-learn；40 位數的 C_llr 要 mpmath | 1 s |
| `diarization.py` | `der`、`jer`、`detection_error_rate` | pyannote.metrics；scipy | 4 s |
| `f0eval.py` | F0 指標 | mir_eval | 2 s |
| `separation.py` | `pit_si_sdr`、`si_sdr_improvement`、`bss_eval` | torchmetrics、mir_eval | 70 s；較新的 numpy 上 3 分鐘 |
| `stats.py [--repetitions N]` | `bootstrap_ci`、`paired_ci`、`rate_ci`、`paired_rate_ci`、`permutation_test`、`holm` | scipy | 85 s；`--repetitions 500` 時 30 s |
| `band_limit.py` | `audio_report` 的頻寬上限 | 不需要 | 2 分鐘 |
| `report_checks.py` | `tonal_lines`，以及 `audio_report` 的削波計數和樣本格點 | 不需要 | 2 分鐘 |
| `against_librosa.py [--all] [--long]` | 重新取樣、短時傅立葉轉換與反轉換、梅爾頻帶與 MFCC、靜音切割、`f0`、`time_stretch` | librosa | librosa 0.11 約 1 分鐘、1.0 約 1.5 分鐘；加 `--all --long` 是 6 和 4.5 分鐘 |

這一頁的數字來自五個環境：

- Python 3.9.13、numpy 2.0.2、scipy 1.13.1、librosa 0.11.0、soxr 1.1.0 和 scikit-learn 1.6.1：
  `lsd.py`、`wada.py`、`wer.py`、`verification.py`、`stats.py`、`siib.py`、`band_limit.py` 和
  `report_checks.py`，以及 `quality.py` 和 `srmr.py` 中不需要參考套件的那些列。
- Python 3.10.22、numpy 2.2.6、scipy 1.15.2、librosa 0.11.0、pysepm_evo 0.1.1
  （搭配 numba 0.67.0）、pesq 0.0.4、SRMRpy 1.0（搭配 Gammatone 1.0.3）、
  commit 226c2f3 的 pySIIB 和 scikit-learn 1.7.2：除了 `wer.py`、`f0eval.py`、`separation.py`、
  `diarization.py`、`dnsmos.py` 和 `stats.py` 以外的每個腳本；`melcepstrum.py` 另用從原始碼
  編譯的 pysptk 1.0.1 和 fastdtw 0.3.4。
- Python 3.13.9、numpy 2.3.5、scipy 1.16.3、librosa 1.0.0、mir_eval 0.8.2、
  scikit-learn 1.7.2、mpmath 1.3.0、onnxruntime 1.32.0，以及 torch 2.14.1 上的
  torchmetrics 1.9.0：`lsd.py`、`f0eval.py`、`separation.py`、`calibration.py`、
  `dnsmos.py`，以及 `quality.py` 中 60 位數的那幾列。
- Python 3.13.9、numpy 2.5.4、scipy 1.18.1 和 pyannote.metrics 4.2.0：`diarization.py`，
  以及再跑一次的 `stats.py` 和 `siib.py` 的預設執行。
- Python 3.9.13，加上 DNS-Challenge 的 `requirements.txt` 釘的 numpy 1.22.4、librosa 0.8.1
  和 onnxruntime 1.13.1，以及 scipy 1.9.3：`dnsmos.py` 的參考那一半。

`against_librosa.py` 用的是 Python 3.9.13 上的 librosa 0.11.0 和 Python 3.13.9 上的
librosa 1.0.1。時間是在一台桌上型電腦的 CPU 上量的。

## 輸入規則和不載入選用套件的匯入 {#the-input-rule-and-the-import-without-optional-packages}

`tests/test_metric_policy.py` 把[輸入規則](guide/metrics.md#input-a-metric-cannot-score)
寫成兩份案例清單。每個格式不對的案例，例如指標沒有定義的取樣率、兩個聲道、NaN 樣本，
或不是布林值的標籤，都必須引發 `ValueError`，訊息以 `<function>(): ` 開頭。每個格式
正確、卻沒有東西可評的案例，例如空的一對、比一個音框還短的片段、靜音的參考訊號，或
只有一種試驗的清單，都必須得到 nan 和剛好一則 `UserWarning`，格式為
`<function>(): ...; scored nan`，而且警告所屬的檔案就是測試檔本身：警告指向呼叫者的
那一行。另一個測試呼叫 `score`，檢查它的警告指向呼叫 `score` 的那一行。`pesq` 用一個替代 `pesq` 套件的模組來檢查規則，因為 CI 不編譯
這個套件。`tests/test_evaluator.py` 在 40 個隨機配對上，照公式重算 `snr`、`si_sdr`、
`segmental_snr` 和 `sdi`，要求數字到最後一個位元都相同，所以這條規則在有東西可評的
配對上不改變任何結果。

[^hl08]: Y. Hu and P. C. Loizou, "Evaluation of objective quality measures for speech enhancement", IEEE Trans. Audio, Speech, Lang. Process. 16(1), pp. 229-238, 2008, doi:10.1109/TASL.2007.911054.
[^hl06]: Y. Hu and P. C. Loizou, "Evaluation of objective measures for speech enhancement", Proc. Interspeech 2006, doi:10.21437/Interspeech.2006-84.
[^kit88]: N. Kitawaki, H. Nagabuchi and K. Itoh, "Objective quality evaluation for low-bit-rate speech coding systems", IEEE J. Sel. Areas Commun. 6(2), pp. 242-248, 1988, doi:10.1109/49.601.
[^klatt82]: D. Klatt, "Prediction of perceived phonetic distance from critical-band spectra: a first step", Proc. ICASSP 1982, vol. 7, pp. 1278-1281, doi:10.1109/ICASSP.1982.1171512.
[^urgent]: K. Saijo, W. Zhang, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, Y. Fu, W. Wang, T. Fingscheidt and S. Watanabe, "Interspeech 2025 URGENT speech enhancement challenge", Proc. Interspeech 2025, pp. 858-862, doi:10.21437/Interspeech.2025-1363. Code: <https://github.com/urgent-challenge/urgent2025_challenge>.
[^nvsr]: H. Liu, W. Choi, X. Liu, Q. Kong, Q. Tian and D. Wang, "Neural vocoder is all you need for speech super-resolution", Proc. Interspeech 2022, pp. 4227-4231, doi:10.21437/Interspeech.2022-11017, Eq. 7. Code: <https://github.com/haoheliu/ssr_eval>.
[^srmr]: T. H. Falk, C. Zheng and W.-Y. Chan, "A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech", IEEE Trans. Audio, Speech, Lang. Process. 18(7), pp. 1766-1774, 2010, doi:10.1109/TASL.2010.2052247.
[^srmrnorm]: J. F. Santos, M. Senoussaoui and T. H. Falk, "An improved non-intrusive intelligibility metric for noisy and reverberant speech", Proc. IWAENC 2014, pp. 55-59, doi:10.1109/IWAENC.2014.6953337.
[^wada]: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation based on waveform amplitude distribution analysis", Proc. Interspeech 2008, pp. 2598-2601, doi:10.21437/Interspeech.2008-644.
[^mhl09]: J. Ma, Y. Hu and P. C. Loizou, "Objective measures for predicting speech intelligibility in noisy conditions based on new band-importance functions", J. Acoust. Soc. Am. 125(5), pp. 3387-3405, 2009, doi:10.1121/1.3097493.
[^ka05]: J. M. Kates and K. H. Arehart, "Coherence and the speech intelligibility index", J. Acoust. Soc. Am. 117(4), pp. 2224-2237, 2005, doi:10.1121/1.1862575.
[^l13]: P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013, doi:10.1201/b14529. The MATLAB code that comes with it holds the comp_*.m files, composite.m, NCM.m and CSII.m. composite.m alone is also in composite.zip on the author's page, <https://ecs.utdallas.edu/loizou/speech/composite.zip>.
[^ansi]: ANSI S3.5-1997, "American National Standard Methods for Calculation of the Speech Intelligibility Index", American National Standards Institute, New York, 1997.
[^siir]: G. R. Warnes and M. Shaver, "SII: Calculate ANSI/ASA S3.5-1997 (R2024) Speech Intelligibility Index", R package version 1.3.0, 2026, https://CRAN.R-project.org/package=SII.
[^siic]: C. Pavlovic, SII.C, in the C program developer's kit on the site of ASA working group S3-79, http://sii.to/programs.html.
[^jm]: D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed., draft of 19 August 2026: Ch. 2, "Words and Tokens", Sec. 2.9, "Minimum Edit Distance", pp. 28-32; Ch. 16, "Automatic Speech Recognition", Sec. 16.6, "ASR Evaluation: Word Error Rate", pp. 21-24. <https://web.stanford.edu/~jurafsky/slp3/>
[^det]: A. Martin, G. Doddington, T. Kamm, M. Ordowski and M. Przybocki, "The DET curve in assessment of detection task performance", Proc. Eurospeech 1997, pp. 1895-1898, doi:10.21437/Eurospeech.1997-504.
[^bosaris]: N. Brümmer and E. de Villiers, "The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF", NIST SRE'11 Analysis Workshop, Atlanta, December 2011. arXiv:1304.2865.
[^nist]: NIST, "The NIST Year 2008 Speaker Recognition Evaluation Plan" (3 April 2008); "The NIST Year 2010 Speaker Recognition Evaluation Plan" (version r6, 21 April 2010); "NIST 2016 Speaker Recognition Evaluation Plan" (4 August 2016); "NIST 2019 Speaker Recognition Evaluation: CTS Challenge" (22 July 2019); "NIST 2019 Speaker Recognition Evaluation Plan" (16 August 2019, the audio-visual evaluation).
[^ca09]: W. Chu and A. Alwan, "Reducing F0 frame error of F0 tracking algorithms under noisy conditions with an unvoiced/voiced classification frontend", Proc. ICASSP 2009, pp. 3969-3972, doi:10.1109/ICASSP.2009.4960497.
[^da11]: T. Drugman and A. Alwan, "Joint robust voicing detection and pitch estimation based on residual harmonics", Proc. Interspeech 2011, pp. 1973-1976, doi:10.21437/Interspeech.2011-519.
[^r76]: L. R. Rabiner, M. J. Cheng, A. E. Rosenberg and C. A. McGonegal, "A comparative performance study of several pitch detection algorithms", IEEE Trans. Acoust., Speech, Signal Process. 24(5), pp. 399-418, 1976, doi:10.1109/TASSP.1976.1162846.
[^mireval]: C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang and D. P. W. Ellis, "mir_eval: a transparent implementation of common MIR metrics", Proc. ISMIR 2014, pp. 367-372, doi:10.5281/zenodo.1416528.
[^bss]: E. Vincent, R. Gribonval and C. Févotte, "Performance measurement in blind audio source separation", IEEE Trans. Audio, Speech, Lang. Process. 14(4), pp. 1462-1469, 2006, doi:10.1109/TSA.2005.858005.
[^pystoi]: M. Pariente, pystoi, <https://github.com/mpariente/pystoi>, under the MIT licence; STOI after C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, IEEE Trans. Audio, Speech, Lang. Process. 19(7), pp. 2125-2136, 2011, and ESTOI after J. Jensen and C. H. Taal, IEEE/ACM Trans. Audio, Speech, Lang. Process. 24(11), pp. 2009-2022, 2016.
[^upit]: M. Kolbæk, D. Yu, Z.-H. Tan and J. Jensen, "Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks", IEEE/ACM Trans. Audio, Speech, Lang. Process. 25(10), pp. 1901-1913, 2017, doi:10.1109/TASLP.2017.2726762.
[^convtasnet]: Y. Luo and N. Mesgarani, "Conv-TasNet: surpassing ideal time-frequency magnitude masking for speech separation", IEEE/ACM Trans. Audio, Speech, Lang. Process. 27(8), pp. 1256-1266, 2019, doi:10.1109/TASLP.2019.2915167.
[^bd06]: N. Brümmer and J. du Preez, "Application-independent evaluation of speaker detection", Computer Speech and Language 20(2-3), pp. 230-275, 2006, doi:10.1016/j.csl.2005.08.001.
[^ftki92]: T. Fukada, K. Tokuda, T. Kobayashi and S. Imai, "An adaptive algorithm for mel-cepstral analysis of speech", Proc. ICASSP 1992, vol. 1, pp. 137-140, doi:10.1109/ICASSP.1992.225953.
[^sc07]: S. Salvador and P. Chan, "Toward accurate dynamic time warping in linear time and space", Intelligent Data Analysis 11(5), pp. 561-580, 2007, doi:10.3233/IDA-2007-11508.
[^p862]: ITU-T Recommendation P.862 (02/2001), "Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment of narrow-band telephone networks and speech codecs"; P.862.1 (11/2003), "Mapping function for transforming P.862 raw result scores to MOS-LQO"; P.862.2 (11/2007), "Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs". P.862 Amendment 2 (11/2005), "Revised Annex A – Reference implementations and conformance testing for ITU-T Recs P.862, P.862.1 and P.862.2", whose A.2 sets the sampling rates and whose C code, version 2.0, is the reference program; P.862 Corrigendum 2 (03/2018). ITU withdrew P.862, P.862.1, P.862.2 and P.862.3 on 5 January 2024 and points to P.863 instead.
[^dnsmos835]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2022, pp. 886-890, doi:10.1109/ICASSP43922.2022.9746108.
[^dnsmos808]: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", Proc. ICASSP 2021, pp. 6493-6497, doi:10.1109/ICASSP39728.2021.9414878.
[^e87]: B. Efron, "Better bootstrap confidence intervals", J. Amer. Statist. Assoc. 82(397), pp. 171-185, 1987, doi:10.1080/01621459.1987.10478410.
[^et93]: B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993, doi:10.1007/978-1-4899-4541-9.
[^fw07]: C. A. Field and A. H. Welsh, "Bootstrapping clustered data", J. R. Statist. Soc. B 69(3), pp. 369-390, 2007, doi:10.1111/j.1467-9868.2007.00593.x.
[^bn04]: M. Bisani and H. Ney, "Bootstrap estimates for confidence intervals in ASR performance evaluation", Proc. ICASSP 2004, vol. 1, pp. I-409-412, doi:10.1109/ICASSP.2004.1326009.
[^ps10]: B. Phipson and G. K. Smyth, "Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn", Stat. Appl. Genet. Mol. Biol. 9(1), Article 39, 2010, doi:10.2202/1544-6115.1585.
[^h79]: S. Holm, "A simple sequentially rejective multiple test procedure", Scand. J. Statist. 6, pp. 65-70, 1979.
[^vkh18a]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An instrumental intelligibility metric based on information theory", IEEE Signal Process. Lett. 25(1), pp. 115-119, 2018, doi:10.1109/LSP.2017.2774250.
[^vkh18b]: S. Van Kuyk, W. B. Kleijn and R. C. Hendriks, "An evaluation of intrusive instrumental intelligibility metrics", IEEE/ACM Trans. Audio, Speech, Lang. Process. 26(11), pp. 2153-2166, 2018, doi:10.1109/TASLP.2018.2856374.
[^ksg04]: A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual information", Phys. Rev. E 69(6), 066138, 2004, doi:10.1103/PhysRevE.69.066138.
[^librosa]: B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg and O. Nieto, "librosa: Audio and music signal analysis in Python", Proc. 14th Python in Science Conference (SciPy 2015), pp. 18-24, doi:10.25080/Majora-7b98e3ed-003. Code: <https://github.com/librosa/librosa>, under the ISC licence.
[^pywt]: G. R. Lee, R. Gommers, F. Wasilewski, K. Wohlfahrt and A. O'Leary, "PyWavelets: A Python package for wavelet analysis", Journal of Open Source Software 4(36), 1237, 2019, doi:10.21105/joss.01237. Code: <https://github.com/PyWavelets/pywt>, under the MIT licence.
[^gl84]: D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform", IEEE Trans. Acoust., Speech, Signal Process. 32(2), pp. 236-243, 1984, doi:10.1109/TASSP.1984.1164317.
[^s98]: M. Slaney, "Auditory Toolbox, version 2", Technical Report 1998-010, Interval Research Corporation, 1998.
[^dm80]: S. B. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", IEEE Trans. Acoust., Speech, Signal Process. 28(4), pp. 357-366, 1980, doi:10.1109/TASSP.1980.1163420.
[^yin]: A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music", J. Acoust. Soc. Am. 111(4), pp. 1917-1930, 2002, doi:10.1121/1.1458024.
[^pyin]: M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", Proc. ICASSP 2014, pp. 659-663, doi:10.1109/ICASSP.2014.6853678.
[^fg66]: J. L. Flanagan and R. M. Golden, "Phase vocoder", Bell System Technical Journal 45(9), pp. 1493-1509, 1966, doi:10.1002/j.1538-7305.1966.tb01706.x.
[^ld99]: J. Laroche and M. Dolson, "Improved phase vocoder time-scale modification of audio", IEEE Trans. Speech Audio Process. 7(3), pp. 323-332, 1999, doi:10.1109/89.759041.
