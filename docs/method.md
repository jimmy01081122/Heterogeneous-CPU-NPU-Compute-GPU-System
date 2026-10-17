# 面向極小型語言模型推論之 CPU–NPU–Compute GPU 異質處理器架構設計與協同最佳化

**英文題目：** Design and Hardware–Software Co-design of a Heterogeneous CPU–NPU–Compute GPU System for Tiny LLM Inference on PYNQ-Z2

**研究平台：** PYNQ-Z2（Zynq XC7Z020）
vivado 由我手動操作，我會參與優化方向選擇、策略調整、工作流程

**workspace** /home/a/Heterogeneous-CPU-NPU-Compute-GPU-System

**repo** git@github.com:jimmy01081122/Heterogeneous-CPU-NPU-Compute-GPU-System.git

## 一、研究動機與目的

大型語言模型（Large Language Model, LLM）的推論包含矩陣乘法、向量運算、正規化、非線性函數與記憶體存取等不同類型的工作負載。由於各種運算具有不同的平行性及資料存取特性，使用單一處理器架構未必能有效利用有限的硬體資源。

本研究預計使用 PYNQ-Z2 FPGA 開發板，建立由 CPU、NPU 與 Compute GPU 組成的異質處理器系統，探討不同運算核心在 LLM 推論過程中的協同執行方式。

研究將以實際運行現成開源 Tiny LLM 為主要應用，先完成可運作的系統原型，再依據實際效能分析結果進行 Hardware–Software Co-design。

## 二、研究目標

1. 設計具備 INT8 矩陣及向量乘加能力的 NPU。
2. 設計最小可程式化 Compute GPU，以平行執行向量與歸約運算。
3. 整合 ARM CPU、NPU 與 GPU，建立異質運算執行環境。
4. 在 PYNQ-Z2 上執行現成預訓練 Tiny LLM。
5. 利用 HDMI 輸出模型生成內容及系統效能資訊。
6. 比較不同處理器組合的效能，分析資料傳輸、同步與資源使用瓶頸。
7. 根據量測結果最佳化處理器架構與運算分工。

## 三、研究平台與模型

### 3.1 硬體平台

採用 PYNQ-Z2，其主要硬體資源包含：

- 雙核心 ARM Cortex-A9 CPU。
- Xilinx Zynq XC7Z020 FPGA。
- 53,200 LUT、220 DSP48E1、140 BRAM36。
- 512 MB DDR3。
- HDMI 輸出介面。

CPU 使用既有 ARM 處理器；NPU 與 Compute GPU 於 FPGA Programmable Logic 中自行設計。

### 3.2 開源模型

採用 Karpathy 的 `llama2.c` 作為主要推論框架，使用 TinyStories 預訓練模型：

- **stories260K：** 初期模型載入、推論與硬體功能驗證。
- **stories15M：** 主要系統整合及效能測試模型。

使用現成模型權重，不將模型訓練列為主要研究工作。

### 3.3 軟體與開發工具

- Vivado：硬體整合、合成與實作。
- Verilog／SystemVerilog：NPU、GPU RTL 設計。
- C／C++：LLM Runtime、Driver、Scheduler。
- Python：模型轉換、測試資料與效能分析。
- PYNQ Linux：板載執行環境。

## 四、異質處理器系統架構

### 4.1 CPU

CPU 負責 LLM Runtime、Tokenizer、推論控制、工作排程與加速器操作。

此外，CPU 將提供 GPU 與 NPU 尚未支援運算的軟體備援。

### 4.2 NPU

NPU 採用 INT8 MAC Array，初版以 8×8 運算單元為候選架構，主要負責：

- GEMV／GEMM。
- Q、K、V Linear Projection。
- Attention 相關矩陣運算。
- Feed Forward Network 的線性層。

考量 LLM 逐 Token 解碼特性，將優先支援 Matrix–Vector Multiplication。

### 4.3 Compute GPU

GPU 採用最小可程式化 SIMD／SIMT 架構，初版評估 4 個平行運算 Lane。

主要元件包括：

- Instruction Decoder。
- Parallel Execution Controller。
- Register File。
- Arithmetic Processing Lanes。
- Scratchpad Memory。
- Load／Store Unit。

GPU 將針對 RMSNorm、RoPE、Softmax、SiLU 等運算開發向量 Kernel。

研究重點在於可程式化平行運算，而非傳統圖形渲染功能。

### 4.4 記憶體與互連

使用 AXI 與 DMA 進行 CPU、NPU、GPU 之間的資料交換，透過外部 DDR3 儲存模型權重、Activation、KV Cache 與 Framebuffer。

研究將分析不同運算單元之間的資料搬移與同步成本，並探討 Buffer Reuse、Operator Fusion 及工作排程最佳化。

### 4.5 HDMI 顯示系統

使用既有 HDMI IP 作為顯示管線的設計基礎，初期設定 640×480 解析度。

顯示內容包括模型生成文字、推論速度與各運算單元效能資訊。

HDMI Display Engine 與 Compute GPU 為兩個獨立模組。

## 五、研究方法

### 第一階段：CPU-only Prototype

移植 `llama2.c` 至 PYNQ-Z2，完成模型載入、Tokenization 與端到端推論。

建立 FP32 參考結果與 CPU INT8 效能基準。

### 第二階段：NPU 設計與整合

設計 INT8 GEMV／GEMM 核心、AXI 介面及資料緩衝機制。

完成 RTL 模擬、正確性驗證及 FPGA 合成，再將 LLM 的線性運算移至 NPU。

### 第三階段：Compute GPU 設計

建立 GPU 的指令集、平行運算單元與軟體控制介面。

實作並驗證適合 GPU 的向量運算 Kernel，測試其執行延遲與資源成本。

### 第四階段：異質架構整合

將 CPU、NPU、GPU 與 HDMI 整合於同一系統，使 LLM 能依據指定的運算映射完成推論。

### 第五階段：Hardware–Software Co-design

透過 Profiling 分析主要瓶頸，進一步研究：

- 運算核心間的工作分配。
- 靜態與動態 Scheduling。
- Operator Fusion。
- 記憶體資料重用。
- DMA 與運算重疊。
- 不同 GPU Lane 與 NPU MAC 配置的效益。

## 六、實驗設計與效能評估

設計五組實驗：

| 實驗 | 執行架構 |
|---|---|
| A | CPU-only FP32 |
| B | CPU-only INT8 |
| C | CPU + NPU |
| D | CPU + Compute GPU |
| E | CPU + NPU + Compute GPU |

各組使用相同模型、輸入、生成長度與適當對齊的數值精度設定。

主要評估指標包括：

- Tokens per Second。
- 每 Token 推論延遲。
- 各運算核心執行時間。
- DMA 與同步成本。
- LUT、FF、DSP 與 BRAM 使用量。
- DDR 記憶體需求與頻寬。
- 推論數值正確性。
- 系統功耗（依量測設備可用性）。

完整三核心架構不預設必然較快，將以實際實驗分析 GPU 是否具有足夠效益。

## 七、工作階段
| phase | 工作內容 |
|---|---|
| 1 | 硬體環境、CPU 推論及 HDMI 初步驗證 |
| 2 | 15M 模型移植、INT8 推論與 Profiling |
| 3 | NPU RTL 設計、模擬及合成 |
| 4 | CPU + NPU 整合 |
| 5 | Compute GPU 設計及 Kernel 驗證 |
| 6 | CPU + NPU + GPU + HDMI 整合 |
| 7 | 效能評估、Co-design 與研究報告 |

上述期程為初步規劃，須依硬體合成與模型移植結果調整。

## 八、預期困難及解決方式

**FPGA 資源有限：** 透過調整 NPU MAC 數量、GPU Lane 數量及 Buffer 容量降低資源需求。

**記憶體頻寬限制：** 研究 DMA 批次化、權重串流與 Buffer Reuse。

**GPU 額外開銷：** 量測 Kernel Dispatch、DMA 與同步成本，保留 CPU fallback 及運算映射調整能力。

**數值精度問題：** 比較 INT8、Fixed-point 與 CPU FP32 參考結果，必要時採混合精度。

**系統整合困難：** 採模組化開發、分階段測試與單一 Bitstream 整合策略。

## 九、預期研究成果

1. 自行設計並完成驗證的 NPU 與 Compute GPU RTL。
2. 可在 PYNQ-Z2 執行 Tiny LLM 的異質處理器原型。
3. 支援 CPU、NPU、GPU 工作分配的推論 Runtime。
4. 可展示生成內容與效能資訊的 HDMI 系統。
5. 不同處理器組合的效能、資源與資料交換分析。
6. 依據 LLM 特性進行 Hardware–Software Co-design 的設計結果。

## 十、預期研究貢獻

本研究的核心貢獻在於：**於資源受限 FPGA 上，設計並實際驗證 CPU、NPU 與可程式化 Compute GPU 的異質運算架構，分析其執行 LLM 時的運算分工、資料交換及硬體資源取捨。**

透過對照實驗，評估 Compute GPU 對 Tiny LLM 推論的實際必要性與效益，並依據工作負載特性提出協同最佳化方案。

## 參考資料

1. PYNQ-Z2 Hardware Specifications — https://www.sparkle-embedded.com.tw/en/PYNQ-Z2
2. AMD/Xilinx PYNQ Base Overlay — https://github.com/Xilinx/PYNQ
3. Karpathy, llama2.c — https://github.com/karpathy/llama2.c
4. TinyStories Pretrained Models — https://huggingface.co/karpathy/tinyllamas