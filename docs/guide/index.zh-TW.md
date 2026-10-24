# 使用指南 {#user-guide}

這份指南照你手上要做的工作來安排，而不是照模組的結構。每一頁說明
這些函式是做什麼用的、該拿哪一個，以及它們會在哪裡出錯，範例都可以
照原樣執行。每個函式的每個參數都在 [API 參考](../reference/index.md)。

## 依工作挑頁面 {#pick-a-page-by-the-job}

| 你想要 | 請看 |
|---|---|
| 讀取、寫入或轉換音訊檔案，一次處理整個資料夾 | [音訊檔案](audio-files.md) |
| 把音訊轉成模型訓練用的矩陣 | [STFT 與特徵](features.md) |
| 在動手處理之前，先知道一段錄音能不能用 | [檢查錄音](inspection.md) |
| 找出哪裡有人在說話、音高多少 | [語音活動與音高](speech.md) |
| 裁切、接合、濾波、淡入淡出或資料擴增 | [編輯與效果](editing.md) |
| 把片段放進房間，或送進電話線路 | [房間與通道](degradation.md) |
| 從一個語音資料夾和一個雜訊資料夾，做出劣化／乾淨的訓練配對 | [建立含雜訊資料集](synthesis.md) |
| 去除雜訊，或在幾種降噪方法之間做選擇 | [降噪](enhancement.md) |
| 拿處理過的音訊和乾淨的參考訊號比對評分 | [品質評估](metrics.md) |
| 在音訊還在進來的時候錄音、監聽或畫圖 | [錄音與即時檢視](live-audio.md) |

## 錄音受損的順序 {#the-order-a-recording-is-damaged-in}

語音要經過一條鏈才會進到檔案裡，有好幾頁講的就是其中一環：

```text
talker -> room -> + noise -> link -> file
```

房間是摺積性的，雜訊是加上去的，而鏈結（電話線路、編解碼器、會掉封包
的網路）則作用在所有進到麥克風的東西上。[房間與通道](degradation.md)
講房間和鏈結，[建立含雜訊資料集](synthesis.md)照這個順序套用整條鏈，
[降噪](enhancement.md)和[品質評估](metrics.md)則是從檔案往回推。

## 每一頁共通的慣例 {#conventions-on-every-page}

- 程式碼都是 `import kudio`，再從最上層呼叫所有東西：寫
  `kudio.file_load`，而不是 `kudio.core.io.file_load`。
- 波形是以 ±1.0 為滿刻度的 float32 NumPy 陣列。單聲道是一維；超過一個
  聲道時是 `(frames, channels)`，也就是 soundfile 用的排列方式。
- 頻譜圖每個音框一列：`(frames, bins)`。
- 凡是有隨機性的地方都可以傳種子，範例也都有傳，所以頁面上顯示的輸出，
  就是你執行時會得到的輸出。
