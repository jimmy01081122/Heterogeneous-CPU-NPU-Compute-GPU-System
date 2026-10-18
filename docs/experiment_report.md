# Heterogeneous CPU–NPU–Compute GPU Experiment & Co-Design Report

## 1. End-to-End Functional Verification (Greedy Decoding, `temp = 0.0`)

| Model | Config | Backend | Tokens | Generated Text | Bit-Exact Match |
|---|---|---|---:|---|---:|
| `stories260K` | Exp A (FP32) | CPU | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | Baseline (FP32) |
| `stories260K_q80` | Exp B (CPU INT8) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp C (CPU + NPU) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp D (CPU + GPU) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp E (CPU + NPU + GPU) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp E (Untiled R=1, RTL) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp E (Fused NPU->GPU Epilogue, RTL) | `rtl` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories260K_q80` | Exp E (ARM Cortex-A9 QEMU + Sim) | `sim (arm)` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One` | PASS (100%) |
| `stories15M` | Exp A (FP32) | CPU | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big` | Baseline (FP32) |
| `stories15M_q80` | Exp B (CPU INT8) | `sim` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big` | PASS (100%) |
| `stories15M_q80` | Exp C (CPU + NPU) | `sim` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big` | PASS (100%) |
| `stories15M_q80` | Exp D (CPU + GPU) | `sim` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big` | PASS (100%) |
| `stories15M_q80` | Exp E (CPU + NPU + GPU) | `sim` | 32 | `Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big` | PASS (100%) |
| `stories15M_q80` | Exp E (Verilated RTL, 8 tok) | `rtl` | 8 | `Once upon a time, there was a` | PASS (100%) |

## 2. Accelerator Hardware Cycle & DMA Telemetry (`stories260K_q80`, 32 Tokens, Verilated RTL)

| Experiment | NPU Cycles/Tok | NPU Est @ 100MHz | GPU Cycles/Tok | GPU Est @ 100MHz | NPU DMA In/Out (KB) | GPU DMA In/Out (KB) |
|---|---:|---:|---:|---:|---:|---:|
| Exp B (CPU INT8) | 0 | 0.000 ms/tok | 0 | 0.000 ms/tok | 0.0 / 0.0 | 0.0 / 0.0 |
| Exp C (CPU + NPU) | 138,215 | 1.382 ms/tok | 0 | 0.000 ms/tok | 8162.9 / 8104.0 | 0.0 / 0.0 |
| Exp D (CPU + GPU) | 0 | 0.000 ms/tok | 12,367 | 0.124 ms/tok | 0.0 / 0.0 | 733.5 / 418.0 |
| Exp E (CPU + NPU + GPU) | 138,215 | 1.382 ms/tok | 12,367 | 0.124 ms/tok | 8162.9 / 8104.0 | 733.5 / 418.0 |
| Exp E (Untiled R=1, RTL) | 194,863 | 1.949 ms/tok | 12,367 | 0.124 ms/tok | 8162.9 / 8104.0 | 733.5 / 418.0 |
| Exp E (Fused NPU->GPU Epilogue, RTL) | 138,215 | 1.382 ms/tok | 230,138 | 2.301 ms/tok | 8162.9 / 8104.0 | 17380.5 / 857.0 |

## 3. Accelerator Hardware Cycle & DMA Telemetry (`stories15M_q80`, `dim=288, hidden=768, layers=6, vocab=32000`)

| Experiment | Backend | Tokens | NPU Cycles/Tok | NPU Est @ 100MHz | GPU Cycles/Tok | GPU Est @ 100MHz | NPU DMA In/Out (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp B (CPU INT8) | `sim` | 32 | 0 | 0.00 ms/tok | 0 | 0.000 ms/tok | 0.0 / 0.0 |
| Exp C (CPU + NPU) | `sim` | 32 | 2,433,906 | 24.34 ms/tok | 0 | 0.000 ms/tok | 463.8 / 57.9 |
| Exp D (CPU + GPU) | `sim` | 32 | 0 | 0.00 ms/tok | 52,774 | 0.528 ms/tok | 0.0 / 0.0 |
| Exp E (CPU + NPU + GPU) | `sim` | 32 | 2,433,906 | 24.34 ms/tok | 52,774 | 0.528 ms/tok | 463.8 / 57.9 |
| Exp E (Verilated RTL, 8 tok) | `rtl` | 8 | 2,433,955 | 24.34 ms/tok | 51,017 | 0.510 ms/tok | 116.0 / 14.5 |

## 4. Key Hardware-Software Co-Design Findings

1. **8-Row Weight Tiling (`tile_rows=8` vs `tile_rows=1`)**:
   - Untiled (`R=1`): `194,863` NPU cycles/token (`62.36 ms` for 32 tokens at 100 MHz).
   - Tiled (`R=8`): `138,215` NPU cycles/token (`44.23 ms` for 32 tokens at 100 MHz) — **29.1% cycle reduction** and **8x fewer activation buffer reads** by broadcasting one 8-byte activation word across all 8 MAC rows.
2. **On-Chip Activation Buffer Reuse**:
   - Loading `xq` once per attention block (`Wq, Wk, Wv` share `xq`) and once per FFN block (`W1, W3` share `xq`) eliminates 3 out of every 7 activation DMA transfers per Transformer layer (42.9% fewer activation H2D transfers).
3. **Verilated RTL vs C ISA/Behavioral Simulator Consistency**:
   - On `stories15M_q80`, Verilated SystemVerilog RTL (`2,433,955` NPU cycles/tok) and the C cycle-accurate simulator (`2,433,906` NPU cycles/tok) agree within **0.002%** while producing bit-identical token sequences.
