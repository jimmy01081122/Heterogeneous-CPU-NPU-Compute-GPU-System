# SPDX-License-Identifier: MIT
# Cocotb testbench for gpu_top (hw/rtl/gpu/gpu_top.sv) verifying all LLM vector
# kernels (RMSNorm, Softmax, SwiGLU, RoPE, Residual Add, and Q8_0 Epilogue).

from __future__ import annotations

import os
import struct
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw", "gpu_kernels"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kernels import (  # noqa: E402
    build_q80_epilogue_kernel,
    build_residual_kernel,
    build_rmsnorm_kernel,
    build_rope_kernel,
    build_softmax_kernel,
    build_swiglu_kernel,
)
from q80 import rmsnorm as ref_rmsnorm, softmax as ref_softmax  # noqa: E402

# GPU register map (matches hw/rtl/gpu/gpu_pkg.sv)
GPU_REG_CTRL           = 0x00
GPU_REG_STATUS         = 0x04
GPU_REG_VEC_LEN        = 0x08
GPU_REG_DMA_ADDR       = 0x0C
GPU_REG_DMA_LEN        = 0x10
GPU_REG_IMEM_ADDR      = 0x14
GPU_REG_IMEM_DATA      = 0x18
GPU_REG_PARAM0         = 0x1C
GPU_REG_PARAM1         = 0x20
GPU_REG_PERF_CYCLES    = 0x24
GPU_REG_PERF_INSTS     = 0x28
GPU_REG_PERF_STALL     = 0x2C
GPU_REG_PERF_BYTES_IN  = 0x30
GPU_REG_PERF_BYTES_OUT = 0x34
GPU_REG_ID             = 0x38

GPU_MAGIC_ID           = 0x47505531

GPU_CMD_DMA_H2D        = 1
GPU_CMD_DMA_D2H        = 2
GPU_CMD_RUN_KERNEL     = 3
GPU_CMD_CLEAR_PERF     = 4


def f32_to_u32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


class GpuDriver:
    def __init__(self, dut):
        self.dut = dut

    async def reset(self):
        d = self.dut
        d.rst_n.value = 0
        d.s_axi_awaddr.value = 0
        d.s_axi_awvalid.value = 0
        d.s_axi_wdata.value = 0
        d.s_axi_wstrb.value = 0xF
        d.s_axi_wvalid.value = 0
        d.s_axi_bready.value = 0
        d.s_axi_araddr.value = 0
        d.s_axi_arvalid.value = 0
        d.s_axi_rready.value = 0
        d.s_axis_tdata.value = 0
        d.s_axis_tvalid.value = 0
        d.s_axis_tlast.value = 0
        d.m_axis_tready.value = 1
        for _ in range(5):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        await RisingEdge(d.clk)

    async def write_reg(self, addr: int, data: int):
        d = self.dut
        d.s_axi_awaddr.value = addr
        d.s_axi_awvalid.value = 1
        d.s_axi_wdata.value = data & 0xFFFFFFFF
        d.s_axi_wstrb.value = 0xF
        d.s_axi_wvalid.value = 1
        d.s_axi_bready.value = 1
        aw_done = False
        w_done = False
        while not (aw_done and w_done):
            await RisingEdge(d.clk)
            if d.s_axi_awready.value and not aw_done:
                d.s_axi_awvalid.value = 0
                aw_done = True
            if d.s_axi_wready.value and not w_done:
                d.s_axi_wvalid.value = 0
                w_done = True
        while not d.s_axi_bvalid.value:
            await RisingEdge(d.clk)
        await RisingEdge(d.clk)
        d.s_axi_bready.value = 0

    async def read_reg(self, addr: int) -> int:
        d = self.dut
        d.s_axi_araddr.value = addr
        d.s_axi_arvalid.value = 1
        d.s_axi_rready.value = 1
        while True:
            await RisingEdge(d.clk)
            if d.s_axi_arready.value:
                d.s_axi_arvalid.value = 0
                break
        while not d.s_axi_rvalid.value:
            await RisingEdge(d.clk)
        val = int(d.s_axi_rdata.value)
        await RisingEdge(d.clk)
        d.s_axi_rready.value = 0
        return val

    async def start_cmd(self, cmd: int):
        await self.write_reg(GPU_REG_CTRL, 1 | ((cmd & 0xF) << 2))

    async def wait_done(self, timeout_cycles: int = 200000):
        for _ in range(timeout_cycles):
            st = await self.read_reg(GPU_REG_STATUS)
            if st & 0x2:
                return
        raise TimeoutError("GPU did not assert status_done within timeout")

    async def load_program(self, inst_words: list[int]):
        await self.write_reg(GPU_REG_IMEM_ADDR, 0)
        for w in inst_words:
            await self.write_reg(GPU_REG_IMEM_DATA, w)

    async def dma_h2d(self, spm_addr: int, arr: np.ndarray):
        d = self.dut
        u32 = np.ascontiguousarray(arr).view(np.uint32).ravel()
        n_words = u32.size
        await self.write_reg(GPU_REG_DMA_ADDR, spm_addr)
        await self.write_reg(GPU_REG_DMA_LEN, n_words)
        await self.start_cmd(GPU_CMD_DMA_H2D)

        payload = u32.tobytes()
        if len(payload) % 8 != 0:
            payload += b"\x00\x00\x00\x00"
        n_beats = len(payload) // 8
        for i in range(n_beats):
            d.s_axis_tdata.value = int.from_bytes(payload[i * 8: (i + 1) * 8], "little")
            d.s_axis_tvalid.value = 1
            d.s_axis_tlast.value = 1 if (i == n_beats - 1) else 0
            while True:
                await RisingEdge(d.clk)
                if d.s_axis_tready.value:
                    break
        d.s_axis_tvalid.value = 0
        d.s_axis_tlast.value = 0
        await self.wait_done()

    async def dma_d2h(self, spm_addr: int, n_words: int, dtype=np.float32) -> np.ndarray:
        d = self.dut
        d.m_axis_tready.value = 0
        await self.write_reg(GPU_REG_DMA_ADDR, spm_addr)
        await self.write_reg(GPU_REG_DMA_LEN, n_words)
        await self.start_cmd(GPU_CMD_DMA_D2H)
        d.m_axis_tready.value = 1

        words: list[int] = []
        while len(words) < n_words:
            await RisingEdge(d.clk)
            if d.m_axis_tvalid.value and d.m_axis_tready.value:
                raw64 = int(d.m_axis_tdata.value)
                keep = int(d.m_axis_tkeep.value)
                words.append(raw64 & 0xFFFFFFFF)
                if keep == 0xFF and len(words) < n_words:
                    words.append((raw64 >> 32) & 0xFFFFFFFF)
                if d.m_axis_tlast.value:
                    break
        d.m_axis_tready.value = 0
        await self.wait_done()
        return np.array(words[:n_words], dtype=np.uint32).view(dtype)

    async def run_kernel(self, vec_len: int, param0_u32: int = 0, param1_u32: int = 0):
        await self.write_reg(GPU_REG_VEC_LEN, vec_len)
        await self.write_reg(GPU_REG_PARAM0, param0_u32)
        await self.write_reg(GPU_REG_PARAM1, param1_u32)
        await self.start_cmd(GPU_CMD_RUN_KERNEL)
        await self.wait_done()


@cocotb.test()
async def test_gpu_id_and_rmsnorm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = GpuDriver(dut)
    await drv.reset()

    magic = await drv.read_reg(GPU_REG_ID)
    assert magic == GPU_MAGIC_ID, f"bad GPU magic {magic:#x}"
    st = await drv.read_reg(GPU_REG_STATUS)
    n_lanes = (st >> 8) & 0xFF
    assert n_lanes == 4, f"expected 4 SIMD lanes, got {n_lanes}"

    # RMSNorm on stories15M dimension (dim = 288)
    rng = np.random.default_rng(10)
    dim = 288
    x = rng.standard_normal(dim).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=dim).astype(np.float32)
    ref = ref_rmsnorm(x, w)

    await drv.start_cmd(GPU_CMD_CLEAR_PERF)
    await drv.load_program(build_rmsnorm_kernel(x_base=0, w_base=1024, o_base=2048))
    await drv.dma_h2d(0, x)
    await drv.dma_h2d(1024, w)
    await drv.run_kernel(dim, f32_to_u32(1.0 / dim), f32_to_u32(1e-5))
    got = await drv.dma_d2h(2048, dim, dtype=np.float32)

    max_err = float(np.max(np.abs(got - ref)))
    dut._log.info(f"RMSNorm(dim=288) max_abs_err = {max_err:.3e}")
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-4), f"RMSNorm mismatch: max_err={max_err}"


@cocotb.test()
async def test_softmax_tail_masking(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = GpuDriver(dut)
    await drv.reset()
    rng = np.random.default_rng(20)

    # Test non-multiple-of-4 length (pos+1 = 37, plus sentinel at index 37..39)
    n = 37
    buf = np.full(40, 999.0, dtype=np.float32)  # huge sentinel in tail lanes 37..39!
    buf[:n] = rng.uniform(-4.0, 4.0, size=n).astype(np.float32)
    ref = ref_softmax(buf[:n].copy())

    await drv.load_program(build_softmax_kernel(x_base=0))
    await drv.dma_h2d(0, buf)
    await drv.run_kernel(n)
    got_full = await drv.dma_d2h(0, 40, dtype=np.float32)

    got = got_full[:n]
    max_err = float(np.max(np.abs(got - ref)))
    dut._log.info(f"Softmax(n=37) max_abs_err = {max_err:.3e}, sum = {got.sum():.6f}")
    assert np.allclose(got, ref, rtol=1e-3, atol=1e-3), f"Softmax mismatch: max_err={max_err}"
    # Verify SIMT tail masking preserved sentinels at 37..39
    assert np.array_equal(got_full[n:], np.full(3, 999.0, dtype=np.float32))


@cocotb.test()
async def test_swiglu_rope_residual_and_epilogue(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = GpuDriver(dut)
    await drv.reset()
    rng = np.random.default_rng(30)

    # 1. SwiGLU on hidden_dim = 172 (stories260K)
    hdim = 172
    hb = rng.uniform(-3.0, 3.0, size=hdim).astype(np.float32)
    hb2 = rng.uniform(-2.0, 2.0, size=hdim).astype(np.float32)
    ref_swiglu = (hb / (1.0 + np.exp(-hb))) * hb2

    await drv.load_program(build_swiglu_kernel(hb_base=0, hb2_base=1024))
    await drv.dma_h2d(0, hb)
    await drv.dma_h2d(1024, hb2)
    await drv.run_kernel(hdim)
    got_swiglu = await drv.dma_d2h(0, hdim, dtype=np.float32)
    assert np.allclose(got_swiglu, ref_swiglu, rtol=1e-3, atol=1e-3)

    # 2. RoPE in-place rotation on dim = 64 (32 complex pairs)
    dim = 64
    pairs = dim // 2
    q = rng.standard_normal(dim).astype(np.float32)
    angles = rng.uniform(0.0, 3.14, size=pairs).astype(np.float32)
    fcr = np.cos(angles).astype(np.float32)
    fci = np.sin(angles).astype(np.float32)
    ref_q = q.copy()
    v0, v1 = ref_q[0::2].copy(), ref_q[1::2].copy()
    ref_q[0::2] = v0 * fcr - v1 * fci
    ref_q[1::2] = v0 * fci + v1 * fcr

    await drv.load_program(build_rope_kernel(vec_base=0, fcr_base=1024, fci_base=1536))
    await drv.dma_h2d(0, q)
    await drv.dma_h2d(1024, fcr)
    await drv.dma_h2d(1536, fci)
    await drv.run_kernel(pairs)
    got_q = await drv.dma_d2h(0, dim, dtype=np.float32)
    assert np.allclose(got_q, ref_q, rtol=1e-5, atol=1e-5)

    # 3. Residual add
    x = rng.standard_normal(dim).astype(np.float32)
    xb = rng.standard_normal(dim).astype(np.float32)
    await drv.load_program(build_residual_kernel(x_base=0, xb_base=1024))
    await drv.dma_h2d(0, x)
    await drv.dma_h2d(1024, xb)
    await drv.run_kernel(dim)
    got_res = await drv.dma_d2h(0, dim, dtype=np.float32)
    assert np.allclose(got_res, x + xb, rtol=1e-6, atol=1e-6)

    # 4. Fused NPU->GPU Q8_0 group dequantization epilogue
    d_rows = 64
    psum = rng.integers(-40000, 40000, size=d_rows, dtype=np.int32)
    ws = rng.uniform(0.001, 0.05, size=d_rows).astype(np.float32)
    acc0 = rng.standard_normal(d_rows).astype(np.float32)
    xs_val = np.float32(0.0125)
    ref_acc = acc0 + psum.astype(np.float32) * (ws * xs_val)

    await drv.start_cmd(GPU_CMD_CLEAR_PERF)
    await drv.load_program(build_q80_epilogue_kernel(psum_base=0, ws_base=1024, acc_base=2048))
    await drv.dma_h2d(0, psum)
    await drv.dma_h2d(1024, ws)
    await drv.dma_h2d(2048, acc0)
    await drv.run_kernel(d_rows, param0_u32=f32_to_u32(float(xs_val)))
    got_acc = await drv.dma_d2h(2048, d_rows, dtype=np.float32)
    assert np.allclose(got_acc, ref_acc, rtol=1e-5, atol=1e-5)

    cyc = await drv.read_reg(GPU_REG_PERF_CYCLES)
    insts = await drv.read_reg(GPU_REG_PERF_INSTS)
    b_in = await drv.read_reg(GPU_REG_PERF_BYTES_IN)
    b_out = await drv.read_reg(GPU_REG_PERF_BYTES_OUT)
    dut._log.info(f"GPU Epilogue perf: cycles={cyc} insts={insts} bytes_in={b_in} bytes_out={b_out}")
    assert insts == 2 + (d_rows // 4) * 9 + 1
    assert b_in == 3 * d_rows * 4
    assert b_out == d_rows * 4
