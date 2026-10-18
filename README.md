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

## Quick start (Phase 1 & 2, CPU-only + Golden Check)

```bash
tools/download_models.sh                                    # host or container
./dev "python tools/export_q8.py models/stories15M.pt  models/stories15M_q80.bin"
./dev "python tools/export_q8.py models/stories260K.pt models/stories260K_q80.bin"
./dev "make -C sw/runtime"                                  # host + arm, plain + _prof + hetero
./dev "tools/golden.sh check"                               # host & qemu-arm vs tests/golden
```

## RTL Verification & Vivado Synthesis (Phase 3 & 5)

```bash
./dev "make -C hw/tb TARGET=npu"            # NPU 8x8 INT8 MAC array + real 260K/15M GEMV vectors
./dev "make -C hw/tb TARGET=gpu"            # 4-lane SIMD/SIMT Compute GPU + 6 LLM vector kernels
vivado -mode batch -source hw/vivado/synth_check.tcl   # OOC synthesis & utilization/timing reports
vivado -mode batch -source hw/vivado/create_bd.tcl     # Zynq-7020 + NPU + GPU + AXI DMA Block Design
```

## Heterogeneous CPU–NPU–Compute GPU Co-Simulation & Board Execution (Phase 4, 6 & 7)

`run_hetero` (and `run_hetero_prof`) executes all 4 quantized experiment configurations (`B=CPU`, `C=CPU+NPU`, `D=CPU+GPU`, `E=CPU+NPU+GPU`) across 3 interchangeable backends:
- `-b rtl`: In-process Verilated SystemVerilog RTL (`npu_top.sv` + `gpu_top.sv` clocked cycle-by-cycle)
- `-b sim`: Cycle-accurate C behavioral / GPU 32-bit ISA simulator (also runs under `qemu-arm`)
- `-b hw` : PYNQ-Z2 `/dev/mem` + Xilinx AXI DMA + CMA physical buffers

```bash
# Run Experiment E (CPU + NPU + Compute GPU) on Verilated SystemVerilog RTL:
./dev "./sw/runtime/build/host/run_hetero_prof models/stories260K_q80.bin \
        -z models/tok512.bin -t 0.0 -n 32 -i 'Once upon a time' -E E -b rtl -R 8"

# Run full Experiment A-E & Hardware-Software Co-Design Suite:
./dev "python tools/run_experiments.py"
```

## Repository layout

```
docker/                 dev image (Dockerfile, compose.yml, requirements.txt)
hw/rtl/{npu,gpu,common} SystemVerilog sources (fp32_alu, npu_top, gpu_top)
hw/tb/                  cocotb testbenches (test_npu.py, test_gpu.py)
hw/vivado/              Vivado synthesis & Block Design TCL (synth_check.tcl, create_bd.tcl, pynq_z2.xdc)
sw/runtime/src/         instrumented llama2.c runtimes (run_fp32.c, run_int8.c, run_hetero.c, prof.h)
sw/runtime/upstream/    pristine llama2.c @ 350e04f (MIT) for reference/diffing
sw/driver/              accelerator driver & backends (accel.c, accel_rtl.cpp, accel_sim.c, accel_hw.c)
sw/gpu_kernels/         Compute GPU assembler & LLM vector kernels (kernels.py, gpu_kernels_bin.h)
sw/pynq/                PYNQ-Z2 overlay loader & HDMI display runner (hetero_overlay.py)
tools/                  model export, golden model (q80.py), experiment runner (run_experiments.py)
tests/golden/           expected greedy outputs
docs/                   design notes, RTL architecture (rtl_design.md), experiment reports (experiment_report.md)
models/                 (git-ignored) checkpoints, see tools/download_models.sh
```

## Status

| Phase | Content | Status |
|---|---|---|
| 1 | Environment, CPU inference, HDMI bring-up | Done (`Docker`, `QEMU`, `hetero_overlay.py`) |
| 2 | 15M INT8 inference & profiling | Done (`prof.h`, `workload.py`) |
| 3 | NPU RTL | Done (`npu_top.sv` + cocotb 3/3 PASS) |
| 4 | CPU + NPU integration | Done (`run_hetero -E C`, activation reuse + 8-row tiling) |
| 5 | Compute GPU + kernels | Done (`gpu_top.sv` + 6 ISA kernels + cocotb 3/3 PASS) |
| 6 | CPU + NPU + GPU + HDMI | Done (`run_hetero -E E`, `create_bd.tcl`, `accel_hw.c`) |
| 7 | Evaluation & co-design | Done (`run_experiments.py`, `docs/experiment_report.md`) |

## License

`sw/runtime/upstream/*` and derived `sw/runtime/src/run_*.c` are from llama2.c (MIT, © Andrej Karpathy).
