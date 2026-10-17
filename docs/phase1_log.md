# Phase 1 log — environment & CPU reference

## 2026-10-17 — environment + host/qemu golden

**Done**
- Docker dev image `hetero-llm-dev` (Verilator 5.030, Icarus 12, cocotb 1.9.2, arm-linux-gnueabihf-gcc 11.4, qemu-arm 6.2, torch 2.4.1 cpu).
- `llama2.c` @ `350e04f` vendored to `sw/runtime/upstream/`; instrumented copies `sw/runtime/src/run_{fp32,int8}.c` + `prof.h` (14-op taxonomy, `-DPROFILE`, CSV via `$HETERO_PROF_CSV`).
- Q8_0 export via `tools/export_q8.py`:
  - stories15M → GS=32, 17.1 MB
  - stories260K → GS=4, 0.52 MB (**upstream bug**: `export.py` picks GS=64 from `dim` only; `hidden_dim=172` is not a multiple, so `run_int8.c` silently drops the W2 tail. Fixed by choosing GS dividing both.)
- Golden greedy outputs (`tests/golden/`, 128 steps, prompt "Once upon a time"): **host x86 and qemu-arm are bit-identical for all 4 cases** (260K/15M × FP32/INT8).
- NumPy golden model `tools/q80.py` matches the C runtime token-for-token (`tests/test_golden_model.py`), INT32 per-group partial sums exact → used for NPU verification.
- `tools/gen_npu_vectors.py` extracts real GEMV cases; max |psum| per group: 50 277 (260K), 87 961 (15M) → **an INT18 signed group accumulator is sufficient for GS≤32** (worst case 32·128·128 = 524 288 → 20 bits).

**Observations (host x86, not board numbers)**

stories15M INT8 per-op share — the classifier dominates, confirming the feasibility analysis:

| op | share |
|---|---|
| mm_cls (32000×288) | 53.7 % |
| mm_ffn13 | 15.7 % |
| mm_qkv | 9.0 % |
| mm_ffn2 | 8.2 % |
| attention | 6.1 % |
| mm_o | 3.0 % |
| everything else (GPU candidates) | ~4.4 % |

- 15M INT8 diverges from FP32 after ~45 tokens (expected quantization effect, still coherent text). 260K INT8 == FP32 for all 128 tokens.
- 260K with GS=4 has FP32 scales equal in size to the int8 weights (260 KB + 260 KB). For an on-chip-weight NPU this argues for per-row or FP16 scales → feeds decision D5.

**Blocked / next**
- Board not reachable from WSL at 192.168.2.99:22 → on-board experiments A/B pending (`tools/deploy_board.sh baseline`).
- HDMI bring-up needs Vivado + board.
