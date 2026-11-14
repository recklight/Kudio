# 指標的驗證方式 {#how-the-metrics-are-checked}

下表的每個指標，都拿去和算出文獻數字的那份實作比對過；沒有這種實作的，就用第二種
方式把定義重新算一遍來比對。比對用的是 `tools/validation/` 裡的腳本，任何人都可以
重跑。這一頁逐一說明每個指標的定義出自哪裡、拿什麼比對、用了哪些訊號、找到的最大差異，
以及 kudio 刻意和參考實作不同的地方。沒有註明「相對」的差異都是絕對差異。

大部分的比對也收在測試套件裡：測試裡存著參考實作在測試自己的配對上算出的數字，所以
不用安裝參考實作，`pytest` 也會拿這些數字檢查 kudio。例外是 SRMR 工具箱的比值、
PESQ 的一致性測試和 pySIIB 範例圖上的點，因為工具箱的測試錄音、ITU 的一致性錄音和
pySIIB 的範例錄音都沒有收進版本庫。

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

下面各節是細節，最後兩節說明怎麼執行這些腳本，以及輸入規則和不載入 librosa 的匯入
是怎麼測試的。

## Hu 與 Loizou 的指標和綜合指標 {#hu-and-loizous-measures-and-the-composite}

定義出處
:   Hu 和 Loizou 2008：[^hl08] LLR 式 4、IS 式 5、CEP 式 6-7（式 7 是 Kitawaki 等人的 LPC 倒頻譜距離，即他們的式 1[^kit88]）、
    fwSegSNR 式 8-9、WSS 式 10（依 Klatt[^klatt82]）；綜合指標的迴歸式在 Hu 和 Loizou
    2006[^hl06] 第 4.1 節。視窗、eps、音框數和捨入照 Loizou 的 MATLAB 檔案
    （comp_llr.m、comp_is.m、comp_cep.m、comp_wss.m、comp_fwseg.m、composite.m）。[^l13]

比對對象
:   pysepm_evo 0.1.1，這是 pysepm 的一個打包版本，它的品質指標是 Loizou 這些檔案的
    移植：`llr`、`cepstrum_distance`、`wss`、`fwSNRseg`，綜合指標的各部分則是
    `llr(used_for_composite=True)` 和 `SNRseg`。pysepm 沒有 Itakura-Saito，所以腳本裡
    把 comp_is.m 重寫一遍，再用第二種方式檢查：在單位圓上 16384 個點計算兩個 LPC 頻譜
    的 Itakura-Saito 散度。pysepm_evo 0.1.1 自己的 `composite()` 會引發 `NameError`
    （它讀取 `mos_lqo` 和 `pesq_mos`，卻沒有定義它們），所以三個評分是把 pysepm 的各
    部分和 `pesq` 套件 0.0.4 的 PESQ 代入 2006 年的迴歸式來比對，PESQ 的選法和原本的
    pysepm 相同。

訊號
:   `tests/test_quality.py` 的合成語音（諧波聲源經過移動的共振峰形成音節，停頓處是
    低 60 dB 的雜訊底），長 3 s：白雜訊、粉紅雜訊和多人交談雜訊，SNR 為 -5、0、5、10
    和 20 dB；`kudio.rir` 產生的房間，RT60 為 0.3、0.6 和 1.2 s；kudio 的 log-MMSE 對
    5 dB 白雜訊和 0 dB 多人交談雜訊的輸出；在雜訊下於 2 kHz 低通的輸出；以及 16 kHz 下
    4 kHz 以上什麼都沒有的參考訊號。8 kHz 有 21 對，16 kHz 有 22 對，另有 30、50、70
    和 390 個音框的配對，用來檢查捨入。

最大差異
:   相對差異，8 和 16 kHz 分別為：LLR 1.9e-12 和 1.1e-11，CEP 7.5e-13 和 3.9e-12，
    WSS 1.6e-16 和 2.4e-16，fwSegSNR 3.0e-15 和 7.8e-15。IS 對 comp_is.m 為 1.7e-12
    和 4.1e-11，對頻譜散度為 7.7e-13。綜合指標的七個欄位為 1.9e-12 和 3.3e-11，PESQ
    欄位為 0。最大的差異出現在 LPC 擬合接近奇異的地方：多人交談雜訊下的 log-MMSE 輸出，
    以及 16 kHz 下 4 kHz 以上什麼都沒有的參考訊號。

    這些數字的前提是每個頻帶都有一點雜訊。輸出若有一段頻帶是空的、又沒有雜訊填補，
    擬合對 float64 來說就是奇異的（條件數約 1e16），綜合指標裡未封頂的 LLR 會由捨入誤差
    決定。以上面 3 秒的語音在 16 kHz 為例，輸出低通到 1 kHz、不加雜訊：這裡得 6.5993，
    pysepm 得 6.5896，同樣的音框用 60 位數精度算是 6.5800。兩者都在它的 0.3% 以內，
    單一音框最多差 2.97。COVL 會承受這個差的 0.512 倍（這裡是 1.3140，用精確的 LLR
    算是 1.3238）；CSIG 在這裡被限制在 1，CBAK 不含 LLR。有封頂的 `llr`、
    `itakura_saito` 和 `cepstral_distance` 每一位數都一樣，因為這種配對的每個音框都在
    上限。8 kHz 下同樣的低通，和 60 位數精度的相對差異為 7.7e-7。

刻意不同之處
:   LLR、IS、CEP 和 WSS 保留 M 個音框中最好的 `round(0.95 M)` 個，照 MATLAB 的方式
    捨入，0.5 遠離零進位；pysepm 則捨入到偶數，在 30、70 或 390 個音框時少平均一個
    音框。兩者只在這裡分歧，最多差 6.25%（8 kHz 下 30 個音框的 WSS：27.012824 對
    25.424200）；kudio 的音框若以 pysepm 的數量來平均，相對差異在 4.2e-13 以內。
    IS 照論文把每個音框限制在 [0, 100]，comp_is.m 只設上限 100：一段前後補零的語音
    完全複本，在這裡得 0，在 comp_is.m 的範圍下得 -0.3324。它的對數項照 comp_is.m，
    裡面的比值是論文印出的式 5 的倒數。綜合指標的 PESQ 預設在 16 kHz 是寬頻 MOS-LQO，
    在 8 kHz 是窄頻原始分數；見 [CSIG、CBAK 和 COVL](guide/metrics.md#csig-cbak-and-covl)。

## 對數頻譜距離 {#log-spectral-distance}

定義出處
:   URGENT 2025 評分腳本裡的 `lsd_metric`[^urgent]
    （`evaluation_metrics/calculate_intrusive_se_metrics.py`，最後一次修改是 2024-11-19
    的 commit 93e16aa），以及 NVSR 的 `ssr_eval` 裡的 `AudioMetrics.lsd`[^nvsr]
    （`ssr_eval/metrics.py`，最後一次修改是 2022-03-28 的 commit 907395c），兩者都在
    2026-11-13 讀取。

比對對象
:   腳本裡把兩個定義都在 `librosa.stft` 上重寫，因為兩個腳本都用它算 STFT：用 librosa
    0.11.0 跑一次，再用 librosa 1.0.0 跑一次。

訊號
:   長 3 s 的諧波音節，取樣率 8、11.025、16、22.05、24、32、44.1 和 48 kHz：加上 -5
    到 20 dB SNR 的白雜訊、在取樣率的四分之一和 4 kHz 處低通、音量減半再加雜訊、晚
    1 ms、放進 0.4 s 的房間，以及有 0.5 s 掉成零；每種取樣率 9 或 10 對。16 kHz 下另有
    512 到 4800 個樣本的配對，這個長度範圍的音框數由 librosa 的補邊決定。

最大差異
:   相對差異：用 librosa 0.11.0 時，對 URGENT 的定義為 1.0e-11，對 `ssr_eval` 的定義為
    1.1e-10；用 librosa 1.0.0 時，對 URGENT 的定義為 1.6e-11。最後幾位數會隨執行時的
    numpy 和 scipy 而變：numpy 2.2.6 和 scipy 1.15.2 下是 7.8e-12 和 5.6e-11。

刻意不同之處
:   kudio 以 float64 計算。URGENT 的腳本把檔案讀成 float32，`ssr_eval` 載入檔案時也是，
    所以它們的 STFT 以單精度計算。在全頻帶的輸出上，兩者吻合到 6.3e-7；輸出有空的頻帶
    時就會分開，和 URGENT 的數字差 0.38（48 kHz，在 12 kHz 低通），和 `ssr_eval` 的
    差 0.42（44.1 kHz，在 4 kHz 低通）。先捨入成 16 位元的話，後面這一對吻合到 3.5e-8。
    參考訊號是數位靜音的音框照 URGENT 的做法計入，每個得 `-ln(eps)` = 18.4，並發出
    警告。

## 梅爾倒頻譜失真 {#mel-cepstral-distortion}

定義出處
:   URGENT 2025 評分腳本的 `mcd_utils.py` 和 `mcd_metric`[^urgent]（commit c726558
    和 93e16aa），移植自 ESPnet 的 MCD：SPTK 的 `mcep`，也就是 Fukada 等人的梅爾
    倒頻譜分析，[^ftki92] 以及 FastDTW。[^sc07]

比對對象
:   在 pysptk 1.0.1 和 fastdtw 0.3.4 純 Python 版上重寫的這段程式：每個音框的梅爾
    倒頻譜對 `pysptk.mcep`，每條校正路徑對 `fastdtw`，MCD 對 `mcd_metric`，不做
    增益匹配時對 `calculate`。

訊號
:   Hu 與 Loizou 那幾列所用的語音，3 秒，取樣率 8、16、22.05、24、32、44.1 和
    48 kHz：加白雜訊到 5 dB、音量減半、低通到取樣率的四分之一、0.4 秒的房間，以及
    兩側各 1 秒的數位靜音；共 35 對、25,544 個音框。

最大差異
:   梅爾倒頻譜係數 6.2e-15；校正路徑沒有任何不同；MCD 7.1e-15 dB。

刻意不同之處
:   kudio 用 float64 計算。URGENT 的腳本以 float32 讀檔，在這些配對上會讓它自己的
    MCD 移動最多 1.1e-6 dB。編譯版的 `fastdtw` 在累積成本相同時的取捨，和這裡依循的
    純 Python 版不同。

## SRMR {#srmr}

定義出處
:   Falk、Zheng 和 Chan，[^srmr] 正規化版本則是 Santos、Senoussaoui 和 Falk，[^srmrnorm]
    照作者的 SRMRToolbox 在 `SRMR.m` 裡的算法（github.com/MuSAELab/SRMRToolbox，head
    為 2023-05-23 的 690bdd1）。

比對對象
:   工具箱在 `test/test.mat` 裡為測試錄音 `test/test.wav` 存的四個比值，也就是它自己的
    測試腳本預期 `SRMR.m` 算出的值：6.116783821833282（完整）、5.941239326243227
    （快速）、3.525434591882688（快速、正規化）和 2.300403317120358（正規化）。另外是
    Python 移植版 SRMRpy 1.0（<https://github.com/jfsantos/SRMRpy>），把它的每一項選擇
    一次一項放進 kudio 比對。

訊號
:   `test.wav`（16 kHz 下 0.68 s，從工具箱的 checkout 讀取），分別以 int16 和浮點數
    讀入。另有在低 40 dB 雜訊底上的有聲音節，長 3 s：乾聲、RT60 為 0.3、0.6、1.0 和
    1.5 s 的房間、10 dB 白雜訊下，以及在 1 kHz 低通；8 和 16 kHz 各 7 段，四種模式
    都跑。

最大差異
:   和存的比值相對差 4.5e-13（numpy 2.0.2 下為 2.3e-13）。照原樣的 SRMRpy 在完整模式
    最多差 4.0%，快速模式 4.1%，正規化模式 78%；把它的選擇放進 kudio 之後，相對差
    1.1e-12。

刻意不同之處
:   kudio 照 `SRMR.m`，不照 SRMRpy：第一個 256 ms 視窗從訊號開始前 192 ms 起算，
    Hamming 視窗是對稱的，Hilbert 轉換不補零，快速模式用 MATLAB 的 `specgram`，
    `norm=True` 會把最高的調變頻帶移到 30 Hz。SRMRpy 只設 `norm=True` 時仍停在 128 Hz，
    測試錄音得 3.680，工具箱存的是 2.300。kudio 和工具箱不同的地方：8 和 16 kHz 以外的
    取樣率會被拒絕，工具箱則警告後照樣評分；切掉靜音後比一個視窗還短的片段得 nan；
    片段中間單獨一段長靜音會被切掉，工具箱的 `vad_s.m` 則保留它。工具箱（也就是 kudio）
    和論文本身的寫法有三處不同：它把每條包絡濾成八個頻帶，每 64 ms 取一個視窗，Falk 等人
    則是每 32 ms 取一個 256 ms 視窗做 DFT，再把頻率格分成八組（III-A 節）；`norm=True`
    時它讓頻帶停在 30 Hz，Santos 等人報告的是 4 到 40 Hz（第 4 節）；30 dB 的下限，它從
    各聲道平均後的視窗能量的峰值往下算，Santos 等人則是先對音框平均，再取所有聲道和頻帶
    中的最大值（式 1）。

## WADA-SNR {#wada-snr}

定義出處
:   Kim 和 Stern，[^wada] 式 1、6、8 和 17，語音為形狀參數 0.4 的 gamma 分布，雜訊為
    高斯分布。

比對對象
:   腳本裡用第二種方式計算的模型：`E|a + v|` 用摺疊常態分布的平均，`E ln|a + v|` 用
    digamma 值的 Poisson 混合，再以分段 Gauss-Legendre 求積對 gamma 密度積分；兩端的
    閉合形式；以及在 6 個 SNR 下各跑 4e7 個樣本的蒙地卡羅模擬。還有隨這個方法流傳的
    蒙地卡羅表 `Alpha0.400000.txt`，也就是 snreval 的 `wada_snr.m` 讀的那一份
    （D. Ellis，LabROSA，SNREVAL v0.54，2014-07-01，
    <http://labrosa.ee.columbia.edu/projects/snreval/>）。

訊號
:   曲線不需要音訊：從 -20 到 110 dB 的每個整數 dB，以及 -200 和 300 dB。高斯雜訊中的
    gamma 語音，在 -5、0、10、20 和 30 dB 各 5 段 10 s，另有 40 段只有高斯雜訊的
    10 s 片段。

最大差異
:   讀取 SNR 的統計量，在 -20 到 110 dB 之間差 2.3e-11；和 -200 dB 處的閉合形式差
    4.2e-12；蒙地卡羅模擬在 1.44 個標準誤以內。

刻意不同之處
:   曲線從模型算出，而不是從表上讀；那份表在 -17 dB 處往下掉，模型在那裡則是上升。
    以模型的曲線來讀，表中的值在 -10 到 30 dB 之間最多偏 0.25 dB，50 dB 時偏 0.47 dB，
    100 dB 時偏 7.3 dB。計算前會先去掉平均值。剛好為零的樣本不計入，移植版則把它們
    設成 1e-10 的下限，所以補了 10% 零的 0 dB 片段，在這裡讀成 -0.06 dB，在 snreval
    裡讀成 100 dB。超出 -20 到 100 dB 時結果是 nan 並發出警告，移植版則回傳範圍的端點。

## DNSMOS {#dnsmos}

定義出處
:   DNS-Challenge 發布的三個模型：`DNSMOS/sig_bak_ovr.onnx`（P.835）[^dnsmos835]、
    `pDNSMOS/sig_bak_ovr.onnx`（個人化）和 `DNSMOS/model_v8.onnx`（P.808）[^dnsmos808]，
    取 commit 785ab12 的版本，照 `DNSMOS/dnsmos_local.py`（commit 6412bc5，2023-01-10）
    的方式執行，套件用它的 `requirements.txt` 釘的版本：numpy 1.22.4、librosa 0.8.1 和
    onnxruntime 1.13.1。

比對對象
:   在這些版本下重寫並執行的那個腳本：P.808 模型的輸入逐視窗比對，SIG、BAK、OVRL
    的原始值和換算後的值、個人化模型的三個分數，以及 P.808 分數則逐片段比對。
    `librosa="0.9"` 另外對照 librosa 1.0.0 下的同一個腳本，以及 torchmetrics 1.9.0 的
    `deep_noise_suppression_mean_opinion_score`，它七個分數都會算。

訊號
:   Hu 與 Loizou 那幾列所用的語音，16 kHz，長 2、9.5、12.3 和 15.5 秒：乾淨的、加白雜訊到
    5 dB、加粉紅雜訊到 10 dB、加第二個說話者到 0 dB、0.4 秒的房間、低通到 4 kHz，以及兩側
    各 0.5 秒的數位靜音；共 28 個片段、117 個視窗，存成 64 位元浮點數的 WAV 檔給腳本讀。

最大差異
:   P.808 模型的輸入在兩種 librosa 下都是 0。SIG、BAK 和 OVRL 差 6.2e-7，它們的原始輸出
    差 6.8e-7，個人化的分數差 5.5e-7，P.808 差 3.1e-7。和 torchmetrics 相比，七個都是 0。

刻意不同之處
:   onnxruntime 1.13.1 和 1.31.1 對某些視窗的輸出差一個 float32 的最小單位，而且腳本用
    float32 平均原始輸出和 P.808 分數，kudio 則用 float64。這裡每個片段都是 16 kHz、短於
    17 秒：其他取樣率下，腳本用 resampy 重新取樣，kudio 用 soxr；比較長的片段裡，腳本用
    浮點數算出的視窗結尾少一個樣本時會略過那個視窗，kudio 則保留（見
    [檢查錄音](guide/inspection.md#dnsmos-what-a-listener-would-say)）。P.808 用
    librosa 0.9 的輸入算出的分數，和用 0.8.1 的輸入算出的最多差 0.19。

## PESQ {#pesq}

定義出處
:   ITU-T P.862，加上 P.862.1 對應到 MOS-LQO 的關係和 P.862.2 的寬頻延伸[^p862]。
    P.862 的 Annex A 在 2005 年由 Amendment 2 修訂，其中的 C 程式碼（2.0 版）是建議書的
    一部分，和文字敘述有衝突時以程式碼為準（A.1.1）。

比對對象
:   ITU-T 隨那份程式碼發佈的一致性資料。Annex 的測試 2(b)（A.3.2.2(b)）用
    `pesq(mode="nb", scale="raw")` 為 40 對 8 kHz 的錄音評分，和 C 程式碼給的原始分數
    比較，表上列到小數第三位。kudio 的 `pesq` 是 `pesq` 套件 0.0.4，它編譯 2.0 版的
    程式碼，改成從記憶體讀兩個訊號，錯誤以錯誤碼回傳而不是結束程式；kudio 負責選模式，
    並從套件回傳的 MOS-LQO 經 P.862.1 的對應關係反推原始分數。

訊號
:   Annex 自己的資料：40 對來自真實和模擬 VoIP 連線、延遲沿著錄音變動的錄音，其中
    22 對的兩邊長度不同。

最大差異
:   5.0e-4，也就是表格的捨入。Annex 讓一個實作通過的條件，是除了一對之外都在 0.05
    以內，而且每一對都在 0.5 以內。先把較長的一邊裁到較短那邊的長度，像逐樣本比較的
    指標那樣，分數和表最多會差 0.57，有 8 對超過 0.05、2 對超過 0.5，雖然裁掉的只是
    靜音或低位準雜訊；kudio 給 PESQ 的是完整的兩段。

刻意不同之處
:   寬頻分數是 2005 年程式碼的分數，`pesq` 套件保留了原樣發佈的寬頻輸入濾波器。
    Corrigendum 2（2018）把這個濾波器乘上 0.251188，並指出未更正的程式碼預測出的分數
    平均比聽者低 0.8 MOS；在 ITU 為 Supplement 23 的 1736 對寬頻訊號列的表上，更正後
    的分數平均高 0.73，個別的一對從低 0.08 到高 1.40 都有。Annex 的測試 1(a)、1(b)、
    2(a) 和 4 需要 Supplement 23 的錄音或 ITU 的 G.191 重新取樣程式，ITU 的這兩個檔案
    裡都沒有，所以沒有執行；測試 3 沒有自己的資料。

## NCM 和 CSII {#ncm-and-csii}

定義出處
:   兩者都出自 Ma、Hu 和 Loizou，[^mhl09] CSII 另出自 Kates 和 Arehart，[^ka05] 照
    NCM.m 和 CSII.m 的算法（F. Chen 和 P. C. Loizou，2011 和 2012 年，收在 Loizou
    那本書附的程式碼裡[^l13]）。

比對對象
:   pysepm_evo 0.1.1 的 `ncm` 和 `csii`，是 NCM.m 和 CSII.m 的移植。另外是 NCM.m 和
    CSII.m 所在資料夾裡的範例配對，和它的 readme 印出的值比對。兩者從 ANSI
    S3.5-1997[^ansi] 取來的頻帶中心、重要性數值和頻寬，則和這份標準表格的兩份免費副本
    比對：表 1 和表 B.1 用 CRAN 套件 SII 1.3.0，[^siir] 表 1 另用標準工作小組網站上的
    C 程式。[^siic]

訊號
:   和 Hu 與 Loizou 指標同類的配對：8 kHz 有 21 對，16 kHz 有 22 對，CSII 另外也在
    11.025、22.05、44.1 和 48 kHz 跑。為了品質評估頁上關於直流偏移的說明，另有直流
    偏移為峰值 0.001% 到 5% 的參考訊號，以及帶粉紅雜訊底的參考訊號，取樣率 8 和
    16 kHz。

最大差異
:   NCM：8 kHz 為 8.3e-9，16 kHz 為 2.2e-6，全部來自頻帶濾波器的形式：把 kudio 的
    濾波器換成 pysepm 的之後，是 4.4e-16 和 2.2e-16。CSII：每種取樣率都是 4.4e-16，
    比對時照 CSII.m 的做法替 pysepm 加上 float64 的 eps。在範例配對上，CSII 給出
    readme 的 0.5556、0.2480 和 0.0457，四位小數全對。NCM 給出 0.4152，和 pysepm 相同，
    readme 印的則是 0.4162；最後那 0.001 從哪裡來，我還沒找到。表格：完全相同。21 個
    中心頻率是表 1 的，21 個重要性數值是表 B.1 裡簡短易讀文章（short passages of easy
    reading material）那一欄，CSII 的 16 個頻寬是表 1 最低 16 個頻帶的頻寬；兩份副本的
    表 1 也彼此一致。

刻意不同之處
:   NCM 的頻帶濾波器以二階節串接執行。NCM.m 把每個濾波器當成一個傳遞函數執行，在
    16 kHz 的最低頻帶會偏離精確的濾波器，達到該頻帶峰值的 1.6e-4；二階節則維持在
    2.5e-14 以內。相關係數的平方捨入後超過 1 時，例如完全複本、縮放或反相的複本，會
    限制在 1，pysepm 則回傳 nan；數位靜音的輸出 NCM 得 0，NCM.m 和 pysepm 回傳 nan；
    短到湊不出三個包絡樣本的一對得 nan。CSII 照 CSII.m 在兩個訊號上都加 eps（pysepm
    沒有加），把同調限制在 [0, 1]，某一級沒有音框或只有一個音框時得 nan，CSII.m 則對
    空的那一級回報 0。

## SIIB {#siib}

定義出處
:   Van Kuyk、Kleijn 和 Hendriks：SIIB 出自他們的短文，[^vkh18a] 式 2 和 8 到 12；
    SIIB^Gauss 出自他們對侵入式指標的評估，[^vkh18b] 第 IV-C 節。SIIB 每個通道的資訊量
    用 Kraskov、Stögbauer 和 Grassberger 的第二個估計器計算，[^ksg04] 也就是他們的
    式 9。論文沒有講清楚的細節，或作者的 MATLAB 程式做法不同的地方，kudio 照程式碼，
    也就是 SIIB.m 和 SIIB_Gauss.m；估計器則照 SIIB.m 呼叫的 MIxnyn，Kraskov 的 MILCA
    套件裡的程式。[API 參考][kudio.core.siib]裡這個模組的說明列出每一處，以及它對分數
    的影響。

比對對象
:   `tools/validation/siib.py` 裡用第二種方式重算的兩個指標，寫法求好讀而不求快：
    MATLAB 的重新取樣逐個抽頭加總，頻譜用明寫的 DFT 矩陣，遮蔽是對各個延遲的迴圈，
    堆疊一個一個填，每個通道的資訊量則從全部 N^2 個距離算出。每個階段都在同一份輸入上
    比對，SIIB 比兩次：一次不加打破平手的雜訊，一次走 SIIB.m 的整條流程，包括雜訊、
    MIxnyn 讀的檔案裡的 8 位有效數字和它答案的 8 位小數，用另一個亂數產生器抽。再來是
    Python 移植版 pySIIB（<https://github.com/kamo-naoyuki/pySIIB>，2022-04-04 的
    commit 226c2f3），在同一個腳本裡只透過它的 `SIIB()` 呼叫執行，用同樣的配對，以及
    模組說明裡那組重新取樣的掃描。它預設的窗函數是 `'hanning'`，scipy 1.15 已經不認得
    這個名稱，所以改傳 `'hann'`，也就是這個名稱原本指的窗。另外是 pySIIB 在它的範例裡
    畫出的點：它的 20 kHz 錄音（20.94 s）以語音形狀雜訊混成的 15 個混合，SNR 從
    -22.55 到 +0.31 dB，用 kudio 評分，再和從 pySIIB 發佈的兩張圖上讀出的點比對。圖上
    每個點都畫在整數像素上，所以 SIIB^Gauss 的圖讀得到 ±0.067 bit/s，SIIB 的圖讀得到
    ±0.128。SIIB.m、SIIB_Gauss.m、MILCA 的 MIxnyn 和 pySIIB 都是 GPL-3.0，kudio 裡
    沒有它們的程式碼。pySIIB 是在 kudio 旁邊執行、拿來比對的對象；SIIB.m 需要 MATLAB，
    沒有執行。

訊號
:   第二種方式用的是諧波聲源經過三個移動的共振峰組成的音節，每五個裡有兩個前面帶一段
    嘶聲，底噪比峰值低 54 dB（40 dB 門檻會把它去掉）或低 24 dB（門檻會保留）：加上
    -5、0、5 和 10 dB 的白雜訊、在 RT60 為 0.5 s 的房間裡、在 2 kHz 低通、以 8 kHz
    製作並加上 0 dB 雜訊、以 44.1 和 48 kHz 製作並加上 5 和 10 dB 雜訊，以及淡出到
    -90 dB 後有 3 s 數位靜音的一對；共 10 對、每對 30 s，預設執行則是 4 對、每對
    12 s。第一對另外比對複本、反相的複本、數位靜音的輸出，以及兩個訊號都加上 0.5 的
    偏移。pySIIB 用同樣這些，再加上重新取樣掃描的 21 對：`tests/test_siib.py` 的 36 s
    測試語音，用近乎理想的濾波器做成 20 和 8 kHz、用 `resample_poly` 自己的濾波器再做
    一次 8 kHz，加上 -10 到 20 dB 的白雜訊。pySIIB 的圖則用範例的 15 個混合，門檻保留
    1674 個音框中的 1638 個，也就是 1623 個堆疊。

最大差異
:   到轉換為止逐位元相同，只有頻帶的中心、頻寬和權重相對差 1.9e-14，頻帶功率差
    4.9e-15；重新取樣和 kudio 呼叫的 scipy `resample_poly` 相對差 1.1e-15。特徵值相對
    差 2.6e-14，特徵通道差到該通道標準差的 5.0e-8。SIIB^Gauss 相對差在 1e-12 以下，這裡
    幾次執行介於 1.5e-13 和 4e-13，隨 BLAS 的執行緒數而變。在同樣
    的通道上，kudio 的近鄰搜尋和全部配對的搜尋數出來的點數相同，所以估計器給出逐位元
    相同的數字；每個通道的資訊量和不加雜訊的版本最多差 3.9e-4 位元，SIIB 相對差
    1.4e-5，和用另一個產生器抽的 SIIB.m 流程則差 6.3e-6，這正是打破平手的雜訊的大小
    （見下面）。複本得到上限 1335.76 bit/s，比它多一個 float64 的最小間隔。

    和 pySIIB 在 16 kHz 比，完整執行的 7 對加上預設執行的 2 對：SIIB^Gauss 相對差最多
    8.2e-13，出現在 2 kHz 低通的那一對，其他各對都在 2.2e-13 以下；SIIB 相對差最多
    1.4e-5，也就是打破平手的雜訊。複本和反相的複本逐位元相同，兩個訊號都加上 0.5 的
    偏移時，SIIB^Gauss 差 1.3e-13、SIIB 差 3.5e-6。在數位靜音的輸出上，輸出只剩下對數
    底下的那個下限，kudio 和第二種方式在這裡彼此一致，pySIIB 則和兩者的 SIIB^Gauss 差
    0.8% 到 1.0%（30 s 上是 0.682 對 0.689 bit/s）；這時的 SIIB 大部分是打破平手的
    雜訊。其他取樣率下兩者重新取樣的方式不同，8 kHz 時 SIIB 相對差最多 8.5e-3、
    SIIB^Gauss 1.2e-3，44.1 和 48 kHz 時是 2.4e-3 和 1.5e-4。在重新取樣掃描的 21 對上，
    pySIIB 給出的是更陡那個濾波器的分數，SIIB 差在 8.5e-4 bit/s、SIIB^Gauss 差在
    3.6e-10 以內，和 kudio 則最多差 5.1 和 1.6 bit/s。

    和 pySIIB 的圖相比，SIIB^Gauss 的 15 個點全在讀數範圍內，最多差 0.062 bit/s。SIIB 最多差
    3.2 bit/s：pySIIB 把一對轉到 16 kHz 用的濾波器比 MATLAB 的長，從 20 kHz 是 365 個
    抽頭，MATLAB 的設計是 101 個，在這些混合上讓 SIIB 變動 -3.0 到 +3.3 bit/s，
    SIIB^Gauss 最多 0.03。改用 pySIIB 的濾波器轉到 16 kHz 的混合來評分，SIIB 的 15 個
    點全在讀數範圍內，最多差 0.127，其中最低的兩個點落在同一列像素上，要反過來配對。

刻意不同之處
:   打破平手的雜訊用 numpy 的 PCG64 和固定的種子產生；SIIB.m 從 MATLAB 的全域產生器抽，
    沿用那個工作階段留下的狀態（新開的工作階段第一次呼叫會重複同樣的結果），MIxnyn
    則用資料當自己的種子，所以 kudio 對同一對每次給同一個分數，SIIB.m 每次呼叫則會差到
    十萬分之幾；分數的 1e-4 以下的差異，我都不當一回事。去掉靜音後語音不到 5.45 s
    （436 個音框）的一對，得到 nan 和一則警告，SIIB.m 則給一個隨特徵值分解而變的數字；
    從頭到尾是數位靜音或同一個值的參考訊號也一樣，SIIB.m 會回傳一個小的隨機數，
    SIIB_Gauss.m 則在特徵值分解時停下。長度不同的一對會截成較短的長度並發出警告，超過
    一半就拒絕，和 kudio 每個指標處理一對的方式相同，SIIB.m 則直接拒絕；不是整數赫茲的
    取樣率會被拒絕，低於 4000 Hz 的也會，那比較可能是以 kHz 給的取樣率，而不是一段錄音，
    而且要放大 4 倍以上才到 16 kHz。SIIB^Gauss 裡沒有功率的通道算成不共有任何資訊，SIIB_Gauss.m 則是
    0 除以 0，讓整對得 0。其餘都是捨入：遮蔽曲線寫成 (1 - w) L + w floor、精確的
    digamma、落在 [0, 1) 的均勻雜訊，每一項大約是特徵的 1e-14。16 kHz 以外的取樣率
    照 SIIB.m 的重新取樣，而不是 pySIIB 的。

## WER 和 CER {#wer-and-cer}

定義出處
:   兩個詞序列或字元序列之間的 Levenshtein 距離，每個編輯的代價都是 1；WER 照
    Jurafsky 和 Martin 在這個距離上的定義。[^jm] 語料合併的方式和 Kaldi 的
    `compute-wer`、jiwer 相同。

比對對象
:   腳本裡自己寫、和 kudio 不共用程式碼的計數器：把小配對的每一種對齊都列出來；從兩端
    填滿完整的編輯距離表，在最短路徑中取命中最多的；以及 Jurafsky 和 Martin 的範例
    （第 16.6 節的 CallHome 語句，13 個詞裡 6 個替換、1 個刪除、3 個插入，76.9%；
    它的 NIST 範例；第 2.9 節的「intention」對「execution」）。裝了 nltk 時也和它的
    `edit_distance` 比對（我用的是 3.9.2）。`compute-wer`、sclite 和 jiwer 4.0.0 的
    規則，則是從它們的程式碼或手冊重寫。

訊號
:   列舉所有最多 6 個 token 的配對（3961 組）、最多 300 個 token 的表、從四個詞中抽出、
    最多 12 個詞的 18,420 組隨機配對，以及一個 5,000 句、100,000 詞、大約 10% 的詞有錯
    的模擬語料。

最大差異
:   每個計數都是 0，所以每個比率也是 0。

刻意不同之處
:   在錯誤最少的對齊裡，kudio 取命中最多的那一種。`compute-wer` 照固定的步驟順序決定，
    在 18,420 組短配對中有 618 組拆出的替換、刪除、插入不同，總數相同；在模擬語料的
    語句上則一組都沒有。sclite 的權重在短配對中確定有 59 組算出比最少還多的錯誤，另外
    711 組要看它自己怎麼處理平手。`cer` 預設不算空白；`spaces=True` 算的和 jiwer 在
    單一空格文字上算的一樣，腳本在 1000 組配對上檢查過。詞是 `str.split()` 切出來的，
    jiwer 則以空格字元切分，所以「hello world」對「hello\nworld」在 jiwer 是 1.0，
    在這裡是 0。參考那一邊完全沒有詞時得到 nan，jiwer 4.0 則回傳插入的數量。

## EER、最小 DCF 和 DET 曲線 {#eer-minimum-dcf-and-the-det-curve}

定義出處
:   DET 曲線的操作點；[^det] EER 以相鄰操作點之間的直線來讀，和 VoxSRC 2020-2023 的
    評分程式、NIST SRE 2016 評分腳本（4.1 版，WeSpeaker 收錄的版本）相同，或以凸包
    來讀，也就是 BOSARIS 的 ROCCH-EER[^bosaris]（第 2.6.2 節）；偵測成本和它的正規化
    照 NIST 評測計畫的定義。[^nist]

比對對象
:   一個暴力掃描，在每個不同的分數、每個中點和兩端，逐筆數錯誤，全部以有理數精確計算；
    用 Andrew's monotone chain 以整數計算 ROC 凸包，並把 ROCCH-EER 再算成所有先驗下
    最小成本的最大值（BOSARIS 式 26）；scikit-learn 的 `roc_curve` 和 `det_curve`
    （1.6.1 和 1.7.2）；VoxSRC 評分程式在 `roc_curve` 上用的 `brentq`；以及從程式碼
    重寫的 NIST SRE 2016 腳本、Kaldi 的 `compute-eer` 和 `compute_min_dcf.py`、
    voxceleb_trainer 和 SpeechBrain 的規則。

訊號
:   600 份 2 到 40 筆試驗的清單，分數有連續值、少數幾個整數等級，或捨入到小數一位；
    6 份 1,000 到 10,000 筆的清單，有些含同分；40,000 筆高斯分數的清單；一份 22,000
    筆、分數捨入到標準差十分之一的清單；以及一份 1,003,302 筆、刻意讓凸包有 8830 個
    頂點的清單。

最大差異
:   兩種 EER 和 DET 曲線對精確掃描和凸包都是 0；min DCF 在 8 組成本設定下為 2.2e-16；
    對 VoxSRC 的 `brentq` 為 1.9e-12，也就是它的容許誤差；對 NIST 的腳本和
    `roc_curve` 為 1.1e-16；對 scikit-learn 的 `det_curve` 為 0，它的點是 kudio 那些點
    裡連續的一段；對 SpeechBrain 為 2.7e-8，因為它以 float32 計算。

刻意不同之處
:   同分的試驗算同一個門檻。Kaldi 的 `compute_min_dcf.py` 和 NIST 的腳本逐筆排序試驗，
    所以會在同分的試驗之間放進操作點，而沒有任何門檻能落在那裡。在那 22,000 筆試驗上，
    把同分的非目標試驗排在前面時，Kaldi 的規則在 `p_target` 0.05 下給出 0.39965 的
    min DCF，但沒有任何門檻能低於 0.43615；NIST 的腳本則依同分試驗的順序，把 EER 讀成
    0.05985 或 0.07050，kudio 不論順序都是 0.06535。上面和 NIST 的比對，用的是 291 份
    沒有目標和非目標試驗同分的清單。

## 校準：C_llr、最小 C_llr、實際 DCF 和 C_Primary {#calibration-c_llr-min-c_llr-the-actual-dcf-and-c_primary}

定義出處
:   C_llr 和用 pool-adjacent-violators 擬合算出的最小 C_llr 照 Brümmer 和
    du Preez；[^bd06] 門檻 log(beta) 上的實際成本，以及它在 p_target 0.01 和 0.005
    上的平均 C_Primary，照 NIST SRE 2016 評測計畫的 3.1 節。[^nist]

比對對象
:   C_llr 和 40 位數精度的計算比對（mpmath 1.3.0）。最小 C_llr 和從 PYLLR 程式碼
    （github.com/bsxfan/PYLLR，commit 442b10b）重寫的 PAV 與 `min_cllr` 比對，也和
    scikit-learn 1.7.2 的 `IsotonicRegression`，以及重寫的 BOSARIS `min_cllr.m` 比對。
    `act_dcf` 和以有理數精確計算的計數，以及重寫的 PYLLR `fast_Bayes_error_rate`
    比對，p_target 為 0.001、0.005、0.01、0.05 和 0.5，另加 c_miss 10；`c_primary`
    和兩個成本的平均比對。

訊號
:   240 份 4 到 400 個試驗的清單：從兩個高斯分佈抽出、本身就是它們 LLR 的值，同樣的
    值經過平移和縮放，四捨五入到整數或小數一位、讓目標和非目標試驗同分，以及含有
    +inf 和 -inf 的 LLR。

最大差異
:   C_llr 為 2.2e-16 bit；最小 C_llr 對 PYLLR 的規則和 isotonic 擬合為 3.3e-16 bit；
    實際 DCF 為 2.8e-14；C_Primary 為 0。

刻意不同之處
:   最小 C_llr 裡同分的分數共用一個 LLR，和 PYLLR 相同。BOSARIS 的 `min_cllr.m`
    把同分中的非目標試驗排在前面，讓擬合把同分拆開；在 56 份目標和非目標同分的清單
    上，56 份都讀得比較低，最多低 0.30 bit。沒有這種同分時兩者相差 5.2e-8 bit，來自
    BOSARIS 在第 i 個 LLR 加上的 1e-6 i / N。

## 語者分段標記：DER、JER 和偵測錯誤率 {#diarization-der-jer-and-the-detection-error-rate}

定義出處
:   DER 照 NIST RT-09 評測計畫 6.1 節的定義，[^rt09] 以及 md-eval.pl（第 22 版，dscore
    附的版本）的評分方式：在所有評估的時間上配對語者，之後才在每個參考邊界兩側去掉
    給定秒數的 collar，並視需要去掉重疊語音。JER 照 DIHARD 的評分程式 dscore 的
    算法；[^dihard] 偵測錯誤率照 pyannote.metrics 的 `DetectionErrorRate`。

比對對象
:   pyannote.metrics 4.2.0：沒有 collar 且評分重疊語音時，用它的
    `DiarizationErrorRate`；有 collar 或不評重疊語音時，用它的
    `IdentificationErrorRate`，系統語者先以 md-eval.pl 的配對改名，配對由它的
    Hungarian mapper 在未裁切的標記上求得；偵測錯誤率用它的 `DetectionErrorRate`。
    dscore 的 JER 照它的程式碼在 10 ms 音框上重寫（commit e02f949）。DER 也在 1 ms
    音框上重新計數一次。

訊號
:   300 段 60 秒的錄音：1 到 4 位參考語者、0 到 5 位系統語者，每位最多 25 段、每段
    0.2 到 6 秒，邊界落在 10 ms 的格點上，同一位語者重疊的段落已合併。

最大差異
:   四種設定下的 DER 都是 8.9e-16；錯誤的各項秒數為 0；JER 對 dscore 的規則為
    6.7e-16；偵測錯誤率為 4.4e-16。

刻意不同之處
:   pyannote.metrics 用實際評分的時間配對語者，所以有 collar 或不評重疊語音時，它自己
    的 DER 和 md-eval.pl 不同：在上面 300 段中有 18 到 38 段，最多差 0.29。它的
    `JaccardErrorRate` 依共同時間配對，dscore 則取 Jaccard 錯誤總和最小的配對，兩者在
    37 段不同，最多差 0.037。同一位語者兩段重疊的時間，它會算兩次；md-eval.pl 拒絕
    這種檔案，kudio 則把它們合併。md-eval.pl 本身沒有在這裡執行。

## F0 誤差 {#f0-errors}

定義出處
:   GPE、VDE 和 FFE 照 Chu 和 Alwan[^ca09] 的式 1-5，FPE 照 Drugman 和 Alwan[^da11]
    的定義，四種情況出自 Rabiner 等人，[^r76] RMSE 則是以音分計的
    `1200 log2(est / ref)`。

比對對象
:   mir_eval 0.8.2，[^mireval] 在它的定義和 kudio 重合的地方：VDE 是不限音分容許範圍時
    的 1 - overall accuracy，也從它的 voicing recall 和 false alarm 算一次；GPE 是兩邊
    都有聲的音框上的 1 - raw pitch accuracy，容許範圍設在 20% 落下的位置，+315.6 和
    -386.3 音分；FFE 是在 300 到 400 音分之間沒有誤差的配對上，350 音分時的
    1 - overall accuracy；RMSE 用它的 `hz2cents`。每個指標，包括 FPE，也和純 Python
    寫的逐音框計數比對。

訊號
:   測試存了 mir_eval 數字的那一對（1000 個音框），7 對在可評分邊緣的配對，從單一音框
    到從不同時有聲的軌跡，以及 400 組 1 到 3000 個音框的隨機配對。

最大差異
:   全部 408 對的每個音框計數都是 0；比率 1.1e-16；RMSE 4.5e-13 音分；FPE 6.9e-18。

刻意不同之處
:   20% 是相對於參考值、而且是嚴格大於。FPE 除以音框數，和 Rabiner 等人算標準差的方式
    相同；MATLAB 的 `std` 除以音框數減一，100 個音框時高 0.5%。RMSE 計入大誤差。負值
    會被拒絕，而不是當成 Keele 的不確定音框或 MIREX 的無聲猜測來讀。

## 分離：SI-SDRi、PIT 和 BSS Eval {#separation-si-sdri-pit-and-bss-eval}

定義出處
:   SI-SDR 照 `si_sdr` 的算法，兩個訊號先去掉平均值，和 Conv-TasNet 的 SI-SNR 相同
    （Luo 和 Mesgarani，式 15[^convtasnet]）；配對取所有音源平均 SI-SDR 最高的那一組，
    也就是語句層級 PIT 的準則；[^upit] SDR、SIR 和 SAR 照 BSS Eval；[^bss] 進步量是
    估計的分數減去混合訊號對同一個音源的分數。

比對對象
:   torch 2.14.1 上的 torchmetrics 1.9.0：float64 下 `zero_mean=True` 的
    `scale_invariant_signal_distortion_ratio`，以及用它的
    `permutation_invariant_training`（`mode="speaker-wise"`、`eval_func="max"`）。
    逐一嘗試每一種排列。mir_eval 0.8.2 的 `bss_eval_sources`，分別用它自己選的排列
    和 `pit_si_sdr` 找到的排列，SDRi 則把混合訊號當成每一個估計。

訊號
:   2、3、4 位說話者，各是隨機音高的諧波音節，加上各自的雜訊底；每個估計含有其他
    說話者 10% 到 90% 的成分和雜訊，再隨機打亂順序。每種音源數 40 組，8 kHz、1 秒。
    BSS Eval 每四組取一組。

最大差異
:   SI-SDR 對 torchmetrics 在 1160 個配對上為 8.9e-11 dB；PIT 的平均為 4.0e-13 dB，
    120 組的排列全部相同，也和窮舉搜尋的結果一致；BSS Eval 的 SDR、SIR、SAR 和 SDRi
    為 0，30 組都是 mir_eval 的排列；SI-SDRi 對兩個 SI-SDR 相減為 0。

刻意不同之處
:   torchmetrics 的 SI-SDR 除非 `zero_mean=True`，否則不去掉平均值。估計加上峰值 1%
    的直流偏移時，它的預設值和 kudio 在上面的配對上最多差 0.035 dB。

## 信賴區間與符號翻轉檢定 {#confidence-intervals-and-the-sign-flip-test}

定義出處
:   百分位數區間和 BCa 區間，照 Efron 1987[^e87] 和 Efron 與 Tibshirani 1993[^et93]
    第 14 章，加速度由被重抽樣的單位做 jackknife 估計；整組有放回重抽樣的 cluster
    bootstrap，即 Field 與 Welsh 研究的作法；[^fw07] 錯誤率以總和之比重抽樣，和 Bisani
    與 Ney 處理 WER 的方式相同；[^bn04] 符號翻轉檢定，隨機抽取組合時 p 值用 Phipson 與
    Smyth 的 (b + 1)/(B + 1)；[^ps10] 以及 Holm 的逐步調整。[^h79]

比對對象
:   `scipy.stats.bootstrap`（百分位數、BCa，以及 `paired=True`），用的是同一批重抽樣，
    因為 kudio 抽樣的呼叫和 scipy 一樣，scipy 1.13.1 和 1.18.1 都比過；
    `scipy.stats.permutation_test`（`permutation_type='samples'`），全部符號組合列舉，
    也比隨機抽取的組合；以有理數精確列舉全部符號組合；一份慢速的重寫，逐次走過同一批
    重抽樣，用 `math.fsum` 逐筆加總錯誤數和總數，並從各筆資料重算「每次拿掉一組」的
    jackknife；20 筆資料全部 2**20 種符號組合的精確 p 值；以及在每個會改變決定的門檻
    都跑一次的 Holm 逐步程序。

訊號
:   常態和對數常態的數值，每組 5 到 1000 筆，五個種子；12 到 300 筆的配對差；6 到 18 位
    說話者的類 WER 計數（Poisson 詞數上的二項錯誤數，每位說話者有自己的 Beta 錯誤率）；
    精確檢定用 2 到 16 筆配對，有同分和沒有同分的，分組檢定用 10 位各 2 到 8 筆的說話者，
    另有 300 組各 6 個、以十分之一為單位的正負差值，它們的符號組合在精確算術下和觀察到的
    總和同分，在浮點數下則不一定。模擬每格 2000 組資料：2 到 200 筆的獨立常態和對數常態
    資料，以及每位 10 到 30 段語句、組內相關係數 0.3 的說話者。

最大差異
:   百分位數區間的兩端，對 scipy `bootstrap_distribution` 上的 `np.percentile` 是 0；
    對 scipy 自己的兩端，scipy 1.13.1 是 0，1.18.1 改用 `scipy.stats.quantile` 讀兩端，
    差區間寬度的 1.0e-15。BCa 兩端是寬度的 1.3e-14（1.18.1 為 2.4e-14），配對的是
    3.1e-15（3.6e-15）；分組區間和錯誤率對慢速重寫是寬度的 1.1e-15；每組只有一筆的分組
    對不分組是 0。精確 p 值在 2 到 16 筆上對 scipy 是 0，分組的對 scipy 在各組總和上的
    結果是 0，十分之一差值的 900 個檢定對有理數精確值也全是 0；隨機組合的 p 值落在精確值
    的 1.4 個標準誤以內，和 scipy 自己的隨機 p 值相差 3.1 個標準誤以內。Holm 是 0。

    模擬中，以說話者重抽樣的 95% 區間在 10 位和 30 位說話者時，分別有 89% 和 92% 的資料
    組包含真正的平均，以語句重抽樣只有 52% 和 50%；30 位說話者的合併 WER 是 93% 對 67%。
    這些比例的準確度大約是 1 個百分點（一個標準誤，表中印在每個比例旁邊）。虛無假設成立
    時，以說話者翻轉符號的檢定在 0.05 下拒絕了 5.3% 和 4.6% 的資料組，以語句翻轉則是
    48% 和 46%，10,000 組獨立資料上是 5.1%。

刻意不同之處
:   有些重抽樣剛好每筆資料各含一次，5 筆時佔 3.8%，8 筆時佔 0.24%。它們的平均就是估計值，
    但換個順序加總，最後一位可能不同。scipy 依捨入落在哪邊，把它們算成低於或高於估計值；
    kudio 把它們算成等於估計值。5 筆時這讓 BCa 的兩端最多差到寬度的 7.7%，而 kudio 和以
    `fsum` 加總的重寫相差寬度的 2.1e-16。觀察到的差為 0 時，scipy 的同分容許誤差是觀察值
    的 100 eps 倍，也就是 0，它的單尾 p 值就跟著捨入走；kudio 的容許誤差還考慮了加總本身
    的捨入。總和為 0 的 9 組十分之一差值上，kudio 的 27 個 p 值都和有理數精確值相同，
    scipy 有 9 個單尾 p 值不同。隨機組合的雙尾 p 值，kudio 數絕對值至少一樣大的組合，
    取 (b + 1)/(B + 1)；scipy 則把較小的單尾 p 值加倍。全部組合列舉時兩者相同。

## SNR、SI-SDR、STOI 和 SDR {#snr-si-sdr-stoi-and-sdr}

`snr`、`si_sdr`、`segmental_snr` 和 `sdi` 各是幾行 numpy，定義在
[品質評估](guide/metrics.md#what-each-metric-sees)。`stoi` 和 `estoi` 是 pystoi，`sdr`
是 mir_eval 的 BSS Eval。kudio 把兩者裁到共同的長度再傳下去，所以它們的數字就是這些
套件的數字。

## 執行這些腳本 {#running-the-scripts}

在 checkout 的最上層執行：

```bash
python tools/validation/quality.py
```

每個腳本都匯入它所在 checkout 裡的 kudio，印出每項比對的最大差異和容許誤差，有差異
超出容許誤差時以 1 結束。缺少參考套件的腳本，會說明略過了哪些列，其餘照跑。pysepm
和 pySIIB 都是 GPL-3.0；它們只是在 kudio 旁邊執行、拿來比對的對象，kudio 裡沒有它們的
任何程式碼。

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

這一頁的數字來自五個環境：

- Python 3.9.13、numpy 2.0.2、scipy 1.13.1、librosa 0.11.0 和 scikit-learn 1.6.1：
  `lsd.py`、`wada.py`、`wer.py`、`verification.py`、`stats.py` 和 `siib.py`，以及
  `quality.py` 和 `srmr.py` 中不需要參考套件的那些列。
- Python 3.10.22、numpy 2.2.6、scipy 1.15.2、librosa 0.11.0、pysepm_evo 0.1.1
  （搭配 numba 0.67.0）、pesq 0.0.4、SRMRpy 1.0（搭配 Gammatone 1.0.3）、
  commit 226c2f3 的 pySIIB 和 scikit-learn 1.7.2：除了 `wer.py`、`f0eval.py`、`separation.py`、`diarization.py`、
  `dnsmos.py` 和 `stats.py` 以外的每個腳本；`melcepstrum.py` 另用從原始碼編譯的 pysptk 1.0.1 和 fastdtw 0.3.4。
- Python 3.13.9、numpy 2.3.5、scipy 1.16.3、librosa 1.0.0、mir_eval 0.8.2、
  scikit-learn 1.7.2、mpmath 1.3.0、onnxruntime 1.32.0，以及 torch 2.14.1 上的
  torchmetrics 1.9.0：`lsd.py`、`f0eval.py`、`separation.py`、`calibration.py`、
  `dnsmos.py`，以及 `quality.py` 中 60 位數的那幾列。
- Python 3.13.9、numpy 2.5.4、scipy 1.18.1 和 pyannote.metrics 4.2.0：`diarization.py`，
  以及再跑一次的 `stats.py` 和 `siib.py` 的預設執行。
- Python 3.9.13，加上 DNS-Challenge 的 `requirements.txt` 釘的 numpy 1.22.4、librosa 0.8.1
  和 onnxruntime 1.13.1，以及 scipy 1.9.3：`dnsmos.py` 的參考那一半。

時間是在一台桌上型電腦的 CPU 上量的。

## 輸入規則和不載入 librosa 的匯入 {#the-input-rule-and-the-import-without-librosa}

`tests/test_metric_policy.py` 把[輸入規則](guide/metrics.md#input-a-metric-cannot-score)
寫成兩份案例清單。每個格式不對的案例，例如指標沒有定義的取樣率、兩個聲道、NaN 樣本，
或不是布林值的標籤，都必須引發 `ValueError`，訊息以 `<function>(): ` 開頭。每個格式
正確、卻沒有東西可評的案例，例如空的一對、比一個音框還短的片段、靜音的參考訊號，或
只有一種試驗的清單，都必須得到 nan 和剛好一則 `UserWarning`，格式為
`<function>(): ...; scored nan`，而且警告所屬的檔案就是測試檔本身：警告指向呼叫者的
那一行。另一個測試呼叫 `score`，檢查它的警告指向呼叫 `score` 的那一行。新的指標要把
它的案例加進這兩份清單。`pesq` 用一個替代 `pesq` 套件的模組來檢查規則，因為 CI 不編譯
這個套件。`tests/test_evaluator.py` 在 40 個隨機配對上，照公式重算 `snr`、`si_sdr`、
`segmental_snr` 和 `sdi`，要求數字到最後一個位元都相同，所以這條規則在有東西可評的
配對上不改變任何結果。

`tests/test_metric_input.py` 要求每個接受音訊的指標，對訊號和取樣率採取同樣的規則。
一欄 `(n, 1)` 和一列 `(1, n)` 的分數必須和向量相同；int16、放在 int32 裡的 24 位元
PCM、uint8 和 float32 必須照常評分且沒有警告；複數、布林、文字和日期陣列、樣本中有
`None`、長短不一的列、單一個數字、被遮罩的樣本、兩個聲道、int64 和 uint64，都必須
以函式名稱開頭的訊息拒絕；放在 int32 裡的 int16 樣本和縮小 1e-9 倍的浮點音訊，必須
給出數字和剛好一則警告；以 0 維陣列或 numpy 純量傳入的取樣率，分數必須和整數相同，
而文字、布林值、nan、inf、零、負數、list 或 None 都必須以函式名稱拒絕。有安裝 pystoi
和 mir_eval 時，滿刻度 1e160 倍的 STOI、ESTOI 和 SDR，必須得到 nan 和一則說這一對
太大聲的警告。

`tests/test_lightweight.py` 啟動一個新的 Python，在匯入 kudio 之前，先把 librosa、pesq、
pystoi、mir_eval、pandas、matplotlib、onnxruntime、PyAudio 和 sounddevice 在
`sys.modules` 裡設成 `None`，讓匯入其中任何一個都像缺少套件那樣失敗。在那個行程裡，
每個指標模組都必須能匯入；每個不需要選用套件的指標，在所有警告都當成錯誤的情況下，
都必須給出有限的數字；`score` 必須能跑每個不需要套件的名稱；每個建立在 librosa 上的
功能，都必須引發寫出 librosa 的 `DependencyError`；`kudio report` 必須能跑；而且到最後
librosa 仍然沒有被載入。接著測試在自己的行程裡、librosa 已經載入的情況下，再算一次
其中幾個指標，數字必須相同。`tests/test_package.py` 檢查 `import kudio` 不會載入
matplotlib、pandas、librosa、scipy.signal、PyWavelets 和 tqdm 中的任何一個。另一個
測試在真正的 numba 前面放一個無法匯入的 numba，這時 `kudio.f0` 必須引發帶有 numba
訊息的 `DependencyError`，`kudio pitch` 必須以 2 結束且不印出 traceback。

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
