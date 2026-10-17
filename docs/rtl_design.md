# Phase 3 & 5 — NPU 與 Compute GPU RTL 架構設計與驗證紀錄

## 1. 模組總覽

| 子系統 | 檔案 | 說明 |
|---|---|---|
| **Common ALU** | [`hw/rtl/common/fp32_alu.sv`](../hw/rtl/common/fp32_alu.sv) | 可合成之 IEEE-754 FP32、INT32、Q16.16 定點共用運算單元（支援 `FADD`, `FSUB`, `FMUL`, `FMAX`, `FMIN`, `I2F`, `F2I`, 以及 `FRECIP` $1/x$、`FRSQRT` $1/\sqrt{x}$、`FEXP` $e^x$、`FSILU` $\text{SiLU}(x)$） |
| **NPU Package** | [`hw/rtl/npu/npu_pkg.sv`](../hw/rtl/npu/npu_pkg.sv) | NPU AXI4-Lite 暫存器映射、控制命令、三種輸出模式定義 |
| **NPU MAC Array** | [`hw/rtl/npu/npu_mac_array.sv`](../hw/rtl/npu/npu_mac_array.sv) | 參數化 `N_ROWS × N_LANES`（預設 `8×8 = 64` MACs，可切換 `4×16` 等配置）INT8 有號數乘加陣列與每列加法樹、32-bit group accumulator |
| **NPU Act Buffer** | [`hw/rtl/npu/npu_act_buf.sv`](../hw/rtl/npu/npu_act_buf.sv) | 片上 Activation (`x_q` int8) 與 Group Scale (`x_s` fp32) 緩衝區，支援跨多個矩陣（如 `Wq/Wk/Wv` 或 `W1/W3`）重用輸入向量 |
| **NPU Top** | [`hw/rtl/npu/npu_top.sv`](../hw/rtl/npu/npu_top.sv) | 整合 AXI4-Lite 控制介面、64-bit AXI4-Stream 輸入/輸出、三模式後處理管線與硬體效能計數器 |
| **GPU Package** | [`hw/rtl/gpu/gpu_pkg.sv`](../hw/rtl/gpu/gpu_pkg.sv) | 32-bit SIMD/SIMT 指令集編碼（6-bit opcode, 4-bit `rd/rs1/rs2`, 14-bit `imm14`）與暫存器定義 |
| **GPU Decoder** | [`hw/rtl/gpu/gpu_decoder.sv`](../hw/rtl/gpu/gpu_decoder.sv) | 指令解碼器 |
| **GPU Controller** | [`hw/rtl/gpu/gpu_controller.sv`](../hw/rtl/gpu/gpu_controller.sv) | 平行執行控制器：管理 PC、SIMT `vloop_idx`、尾部遮罩 `lane_active`、跨 Lane 歸約樹（`RADD`, `RMAX`） |
| **GPU Regfile** | [`hw/rtl/gpu/gpu_regfile.sv`](../hw/rtl/gpu/gpu_regfile.sv) | `16 × N_LANES × 32-bit` 向量暫存器檔，支援 per-lane predicate write mask |
| **GPU Lane** | [`hw/rtl/gpu/gpu_lane.sv`](../hw/rtl/gpu/gpu_lane.sv) | 單一 SIMD/SIMT 算術運算通道（預設實例化 4 個平行 Lane） |
| **GPU Scratchpad** | [`hw/rtl/gpu/gpu_scratchpad.sv`](../hw/rtl/gpu/gpu_scratchpad.sv) | `4096 × 32-bit`（16 KB）多通道 Scratchpad Memory，支援平行向量存取、Gather/Scatter 與 64-bit DMA |
| **GPU LSU** | [`hw/rtl/gpu/gpu_lsu.sv`](../hw/rtl/gpu/gpu_lsu.sv) | Load/Store Unit（支援 `VLD`, `VST`, `SLD`, `SST`, `VLD_ABS`, `VST_ABS`） |
| **GPU Top** | [`hw/rtl/gpu/gpu_top.sv`](../hw/rtl/gpu/gpu_top.sv) | Compute GPU 頂層模組，含 256-word IMEM、AXI4-Lite、64-bit AXI4-Stream DMA 與硬體效能計數器 |

---

## 2. NPU 設計重點

1. **三種輸出模式（可由 `REG_CTRL[7:6]` 動態切換，支援 Co-design 比較）：**
   - `NPU_OUT_GROUP_PSUM (0)`：輸出每個 `(row, group)` 的精確 INT32 partial sum（共 `d × (n/GS)` 個 32-bit words），可直接串流 `stories260K_q80.bin` 與 `stories15M_q80.bin` 原始權重。
   - `NPU_OUT_ROW_ACC (1)`：在片上跨所有 group 累加出每列 INT32 內積，每列僅輸出 1 個 32-bit word（共 `d` words）。
   - `NPU_OUT_ROW_FP32 (2)`：在片上讀取預載之 `x_s[g]` 與串流之 `w_s[r, g]`，透過 IEEE-754 FP32 乘加器直接輸出每列 dequantized `float32` 結果（共 `d` floats），將輸出 DMA 資料量降低 `n/GS` 倍。
2. **任意維度與小 Group Size 支援：**
   - 自動處理非 `N_ROWS` 倍數之尾部列（如 `stories260K` 的 `d=172 = 21×8 + 4`）以及 `GS < N_LANES`（如 `stories260K` 的 `GS=4`、奇數 64-bit 長度 `n=172`）。
   - 支援 Tiled 權重模式（`cfg_tile_rows = 8`，8×8 MAC 同時運算並共享單次 `x_buf` 讀取）與 Untiled Row-Major 模式（`cfg_tile_rows = 1`）。

---

## 3. Compute GPU ISA 與向量 Kernel

組譯器與 Kernel 定義於 [`sw/gpu_kernels/kernels.py`](../sw/gpu_kernels/kernels.py)，並自動產生 C header [`sw/gpu_kernels/gpu_kernels_bin.h`](../sw/gpu_kernels/gpu_kernels_bin.h)：

| Kernel | 指令數 | 使用之關鍵指令 | Cocotb 實測最大誤差 |
|---|---|---|---|
| **RMSNorm** (`dim=288`) | 19 | `VLD`, `FMUL`, `RADD` (跨 Lane 歸約), `FRSQRT`, `VST` | `1.574e-5` |
| **Softmax** (`n=37`, 含 SIMT 尾部遮罩驗證) | 20 | `SLD`, `RMAX` (跨 Lane 取大), `FSUB`, `FEXP`, `RADD`, `FRECIP`, `FMUL` | `9.239e-6` (`sum = 1.000000`) |
| **SwiGLU** (`hidden_dim=172`) | 8 | `VLD`, `FSILU`, `FMUL`, `VST` | `< 3e-4` |
| **RoPE** (`dim=64`, in-place 複數旋轉) | 19 | `LD_LANE`, `SHL`, `IADD`, `VLD_ABS`, `VST_ABS`, `FMUL`, `FSUB`, `FADD` | `< 1e-6` |
| **Residual Add** (`dim=64`) | 7 | `VLD`, `FADD`, `VST` | `0.0` (bit-exact) |
| **Q8_0 Epilogue** (`d=64`, NPU→GPU 融合) | 12 | `LD_PARAM`, `VLD`, `I2F`, `FMUL`, `FADD`, `VST` | `< 1e-6` |

---

## 4. 模擬與 Vivado 合成指令

```bash
# 在 Docker 容器內執行 Verilator + cocotb 驗證
./dev "make -C hw/tb TARGET=npu"
./dev "make -C hw/tb TARGET=gpu"

# 在安裝有 Vivado 的機器上手動執行 XC7Z020 OOC 合成與資源/時序報告
vivado -mode batch -source hw/vivado/synth_check.tcl
```
