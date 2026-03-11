# Research Design and Evaluation Records

## 問題定義

讀完一本好書後，讀者往往渴望能與作者對話——針對人生難題請益、獲得啟發。本專案旨在模擬這種體驗：將作者語錄建立為檢索資料庫，讓使用者提問，系統基於資料庫內容生成回應，以達成「與作者交流」的感受。

### 實作進度（截至 2026/3）


已完成 Apology、Meno、Gorgias、Republic 四本書籍分塊，並以 BGE 產生 embedding，透過 Bi-encoder（Top-50）→ Cross-encoder rerank（Top-5）pipeline 完成檢索，使用 Qwen2.5-3B-Instruct 作為 generator 產出關於 retrieved citation 的 value 陳述組成最終回應。

Evaluation 的部分已實作完指標，但尚未人工標注製作對照組，在沒有 ground truth 的情況下，只能說「模型自評結果」，無法判斷系統是否真的有用，因此算是一次流程上的實作，尚未變成 production 版本。

## 研究假設

| 代號 | 假設 | 對應評估 |
|------|------|----------|
| H1 | RAG 系統能生成「奠基於作者觀點」的回應 | A1/A2（citation 有效性）、D1 (LLM-as-a-Judge Faithfulness) |
| H2 | 回應內容與使用者問題具足夠相關性 | C（Q–citation similarity）、D2 (LLM-as-a-Judge Relevancy) |
| H3 | LLM 能在檢索基礎上產出真實且具幫助性的內容 | B1/B2 (citation 多樣性)、D1/D2|

---

## 技術棧

本專案使用的技術與工具如下。

### 檢索（Retrieval）

| 項目 | 技術／設定 |
|------|------------|
| Bi-encoder | **BAAI/bge-base-en-v1.5**（sentence-transformers），768 維，passage 不帶 instruction prefix，query 使用 BGE 建議前綴 |
| Reranker（Cross-encoder） | **BAAI/bge-reranker-base** |
| 流程 | Bi-encoder 掃描全 chunk → Top-50 → Cross-encoder rerank → Top-5 送進 LLM |
| 相似度 | Cosine similarity（embedding 已 normalize） |

### 生成（Generation）

| 項目 | 技術／設定 |
|------|------------|
| LLM | **Qwen/Qwen2.5-3B-Instruct**（Hugging Face） |
| 推論後端 | `transformers`（AutoModelForCausalLM, AutoTokenizer） |
| 量化／裝置 | CUDA：4-bit bitsandbytes（bnb_4bit_quant_type=nf4）；Mac：fp16 + MPS；CPU：fp32 |
| 輸出格式 | 依 passage 順序輸出 JSON 陣列（value 字串），再與 citation 合併成最終回應 |

### 資料集與 Chunk

| 項目 | 說明 |
|------|------|
| 書目 | Apology、Meno、Gorgias、Republic（柏拉圖對話錄） |
| Chunk 設計 | 依書本與行號分塊，無 instruction prefix；embedding 前已 normalize |
| 問題集 | `questions_life.json`，100 題人生／倫理哲學相關問句，用於 evaluation |

---

## 研究紀錄

### 小規模嘗試 - 2026/2/20

**實驗目的**：驗證 LLM 是否能依據檢索到的 citation 產生 grounded 回應，而非憑空臆測。

**操作**：將檢索得到的 citation（書名、行號、文本）一併餵給 LLM，要求針對使用者問題產出價值陳述。

**發現**：若僅將 citation 包在 query 中，模型傾向依賴自身知識回答，而非引用實際文本。因此改為要求模型先輸出 **exact quote**（與原文逐字對應的引用），再基於 user question 與 exact quote 生成價值陳述，藉此強制模型必須 grounded 於檢索內容。

---

### First Run — 2026/2/23

此階段尚未實作 generator 評估，僅進行 retriever 評估。

| 指標 | 結果 |
|------|------|
| A1 Citation existence | 100% |
| A2 Exact quote match | 0% |
| B1 Unique retrieved chunks | 269 |
| B2 Unique sections | 19 |
| C Similarity mean (Q–chunk) | 0.5084 |
| n | 100 |

**發現**

1. **Exact quote match 0% 但 citation existence 100%**：輸入 LLM 的文字經 chunk 處理，邊界與引用範圍不一致。移除 max_chunk 限制並加入明確錯誤處理：若超出 context 長度則回傳 error。
2. **Similarity 提升方向**：採用 Two-stage retrieval（Bi-encoder 掃描全 chunk 取 Top-50 → Cross-encoder rerank → Top-3）送進 LLM。

---

### Second Run — 2026/2/25

| 指標 | 結果 |
|------|------|
| A1 Citation existence | 100% |
| A2 Exact quote match | 100% |
| B1 Unique retrieved chunks | 240 |
| B2 Unique sections | 19 |
| C Similarity mean | ≈ 0.5532 |
| n | 100 |

**發現**

1. max_chunk 移除後，exact quote match 為 100%，且無 chunk 超出 context 長度。
2. **Generator 效率**：每個 citation 獨立呼叫 API 導致延遲。改為一次處理所有 citation，並強化 parsing 邏輯。
3. **Similarity 提升有限**：單一 chunk 抽離後語義往往不完整，問題問的是立場，但 chunk 回傳的是片段對話，難以為抽象或跨段落的問題找到精準對應，建議後續實作 **graph-based** 架構，以捕捉段落與概念間的關聯。
4. **新增 LLM-as-a-Judge 作為 Generator 的評估**：但 RAGAS 較偏向 OpenAI API、錯誤頻繁，故改為自行實作 LLM-as-a-Judge pipeline。

---

### Third Run — 2026/3/10

| 指標 | 結果 |
|------|------|
| A1 Citation existence | 100% |
| A2 Exact quote match | 100% |
| B1 Unique retrieved chunks | 220 |
| B2 Unique sections | 18 |
| C Similarity mean | ≈ 0.5606 |
| D1 LLM-as-a-Judege Relevancy (1–5) mean | 4.16 |
| D2 LLM-as-a-Judege Faithfulness (1–5) mean | 3.94 |
| n | 100 |

**發現**
1. Generator 效率大幅提升，且因 Qwen2.5-3B-Instruct 雖較難完全遵守 query 的格式，但自己也自成系統，因此 parsing 無錯誤。
2. LLM-as-a-Judge（以 Qwen2.5-7B 作為 judge）輸出 relevancy 以及 faithfulness，但因缺乏對照組，結論尚不可考。建議針對 retrieval 與 generator 進行人工標注，方能判斷專案是否具 production 可行性。

---

## 評估指標定義

| 類別 | 指標 | 說明 |
|------|------|------|
| **A. Citation Validity** | A1 Citation Existence | 引用是否能在 chunk index 中找到對應區段 |
| | A2 Exact Quote Match | 模型輸出的 quoted text 是否為 chunk 原文子字串 |
| **B. Diversity** | B1 Unique chunks | 檢索結果的多樣性與分散程度 |
| | B2 Unique citations | 模型實際引用的多樣性與重複使用率 |
| **C. Similarity** | Q–Citation similarity (BGE) | 問題與引用文本的 BGE embedding 餘弦相似度 |
| **D. LLM-as-a-Judge** | Relevancy (1–5)、Faithfulness (1–5) | 以 Qwen2.5-7B 評估回應與問題的相關性及與引用的一致性 |

---

## 建議優化內容

1. **問題集生成優化**：目前問題主要由研究者設計，圍繞四本文本的核心思想，關注對於「人生」的提問（如："What makes a life worth living?"）。建議未來可提升題目的數量與多樣性，並針對問題集進行品質評估（如主題涵蓋度、明確性、難易度分層等），以確保題庫的代表性與有效性。
1. **人工標注**：針對 retrieval（檢索到的 chunk 是否為 relevant）與 generator（回應是否 grounded、有幫助）建立 ground truth，作為對照組，才能驗證自動指標的效度與專案 production 可行性。
2. **Chunk 設計**：單一 chunk 語義往往不完整，問題涉及抽象立場或跨段落概念時難精準對應。建議採 **graph-based** 架構，以捕捉段落與概念間的關聯。