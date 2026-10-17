# Heterogeneous CPU–NPU–Compute GPU System for Tiny LLM Inference (PYNQ-Z2)

面向極小型語言模型推論之 CPU–NPU–Compute GPU 異質處理器架構設計與協同最佳化。

- **Platform:** PYNQ-Z2 (Zynq XC7Z020, dual Cortex-A9 @ 650 MHz, 512 MB DDR3 16-bit)
- **Workload:** [`llama2.c`](https://github.com/karpathy/llama2.c) + TinyStories (`stories260K`, `stories15M`)
- **Plan:** [`docs/method.md`](docs/method.md)

## Development environment (Docker)

Everything except Vivado runs in one container (`docker/Dockerfile`):

| Purpose | Tools |
|---|---|
| RTL simulation | Verilator 5.030, Icarus Verilog 12, cocotb 1.9 |
| ARM build | `arm-linux-gnueabihf-gcc` (Ubuntu 22.04 armhf = PYNQ v3.x userland) |
| ARM functional run | `qemu-arm` (no board needed) |
| Python | numpy, torch-cpu, sentencepiece, pandas, matplotlib |

Vivado is operated manually; the repo only keeps rebuildable TCL / constraints under `hw/vivado/`.

```bash
./dev build                       # build image (first time ~10 min)
./dev                             # interactive shell, repo mounted at /workspace
./dev "make -C sw/runtime"        # one-off command
```

## Quick start (Phase 1, CPU-only)

```bash
tools/download_models.sh                                    # host or container
./dev "python tools/export_q8.py models/stories15M.pt  models/stories15M_q80.bin"
./dev "python tools/export_q8.py models/stories260K.pt models/stories260K_q80.bin"
./dev "make -C sw/runtime"                                  # host + arm, plain + _prof
./dev "tools/golden.sh check"                               # host & qemu-arm vs tests/golden
```

Run with per-operator profiling:

```bash
./dev "HETERO_PROF_CSV=prof.csv sw/runtime/build/host/run_int8_prof \
        models/stories15M_q80.bin -z models/tokenizer.bin -t 0 -n 128"
```

## RTL Simulation & Synthesis (Phase 3 & 5)

```bash
./dev "make -C hw/tb TARGET=npu"            # NPU 8x8 INT8 MAC array + real 260K/15M GEMV vectors
./dev "make -C hw/tb TARGET=gpu"            # 4-lane SIMD/SIMT Compute GPU + RMSNorm/Softmax/SwiGLU/RoPE
vivado -mode batch -source hw/vivado/synth_check.tcl   # manual Vivado XC7Z020 OOC synthesis & reports
```

## Repository layout

```
docker/            dev image (Dockerfile, compose.yml, requirements.txt)
hw/rtl/{npu,gpu,common}   SystemVerilog sources (fp32_alu, npu_top, gpu_top)
hw/tb/             cocotb testbenches (test_npu.py, test_gpu.py)
hw/vivado/         synthesis & block-design TCL (run manually in Vivado)
sw/runtime/src/    instrumented llama2.c runtimes (run_fp32.c, run_int8.c, prof.h)
sw/runtime/upstream/  pristine llama2.c @ 350e04f (MIT) for reference/diffing
sw/driver/         accelerator HAL (accel_hal.h)
sw/gpu_kernels/    Compute GPU assembler & LLM vector kernels (kernels.py, gpu_kernels_bin.h)
tools/             model export, golden model (q80.py), test-vector generation
tests/golden/      expected greedy outputs
docs/              design notes, RTL architecture (rtl_design.md), experiment logs
models/            (git-ignored) checkpoints, see tools/download_models.sh
```

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Environment, CPU inference, HDMI bring-up | Docker + CPU/QEMU golden done (board offline) |
| 2 | 15M INT8 inference & profiling | Host profiling + workload model done |
| 3 | NPU RTL | RTL + cocotb verified (260K & 15M vectors) |
| 4 | CPU + NPU integration | HAL ready |
| 5 | Compute GPU + kernels | RTL + ISA + 6 vector kernels cocotb verified |
| 6 | CPU + NPU + GPU + HDMI | |
| 7 | Evaluation & co-design | |

## License

`sw/runtime/upstream/*` and derived `sw/runtime/src/run_*.c` are from llama2.c (MIT, © Andrej Karpathy).
