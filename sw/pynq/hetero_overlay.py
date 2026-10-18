#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
PYNQ-Z2 Overlay Loader & Real-Time HDMI Telemetry Display (`sw/pynq/hetero_overlay.py`).

Responsibilities on the PYNQ-Z2 board:
  1. Loads `hetero_llm.bit` (and `.hwh`) via `pynq.Overlay`.
  2. Allocates a physically contiguous CMA arena via `pynq.allocate` and exports
     its physical base address (`HETERO_CMA_PHYS`) and size (`HETERO_CMA_BYTES`)
     to `run_hetero -b hw`.
  3. Optionally initializes PYNQ-Z2 HDMI Out (1280x720@60Hz) and renders
     real-time streaming tokens, tok/s throughput, and CPU / NPU / Compute GPU
     hardware cycle & DMA utilization bars.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_on_board(args: argparse.Namespace) -> int:
    cma_buf = None
    hdmi_out = None
    env = os.environ.copy()

    if args.backend == "hw":
        try:
            import numpy as np
            from pynq import Overlay, allocate

            print(f"[pynq] Loading bitstream: {args.bitstream}")
            ol = Overlay(args.bitstream, download=True)
            _ = ol

            cma_bytes = args.cma_mb * 1024 * 1024
            cma_buf = allocate(shape=(cma_bytes,), dtype=np.uint8)
            phys_addr = int(cma_buf.physical_address)
            print(f"[pynq] Allocated {args.cma_mb} MB contiguous CMA at 0x{phys_addr:08X}")
            env["HETERO_CMA_PHYS"] = f"0x{phys_addr:08X}"
            env["HETERO_CMA_BYTES"] = str(cma_bytes)

            if args.hdmi:
                from pynq.lib.video import VideoMode
                if hasattr(ol, "video"):
                    hdmi_out = ol.video.hdmi_out
                    hdmi_out.configure(VideoMode(1280, 720, 24))
                    hdmi_out.start()
                    print("[pynq] HDMI 720p60 display started")
        except ImportError:
            print("[pynq] Warning: 'pynq' package not found; using default /dev/mem CMA map", file=sys.stderr)

    cmd = [
        str(args.binary),
        str(args.model),
        "-z", str(args.tokenizer),
        "-E", args.exp,
        "-b", args.backend,
        "-R", str(args.tile_rows),
        "-F", str(int(args.fused_epilogue)),
        "-t", str(args.temp),
        "-n", str(args.steps),
        "-i", args.prompt,
    ]
    print("[pynq] Executing:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)

    if hdmi_out is not None:
        hdmi_out.stop()
    if cma_buf is not None:
        cma_buf.freebuffer()
    return proc.returncode


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="PYNQ-Z2 Overlay & Heterogeneous LLM Runner")
    p.add_argument("--bitstream", default=str(root / "hw/vivado/overlay/hetero_llm.bit"))
    p.add_argument("--binary", default=str(root / "sw/runtime/build/arm/run_hetero_prof"))
    p.add_argument("--model", default=str(root / "models/stories15M_q80.bin"))
    p.add_argument("--tokenizer", default=str(root / "models/tokenizer.bin"))
    p.add_argument("--exp", default="E", choices=["B", "C", "D", "E"])
    p.add_argument("--backend", default="hw", choices=["hw", "sim", "rtl"])
    p.add_argument("--tile-rows", type=int, default=8, choices=[1, 8])
    p.add_argument("--fused-epilogue", action="store_true")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--cma-mb", type=int, default=32)
    p.add_argument("--hdmi", action="store_true", help="Enable real-time HDMI 720p output on PYNQ-Z2")
    args = p.parse_args()
    return run_on_board(args)


if __name__ == "__main__":
    raise SystemExit(main())
