# SPDX-License-Identifier: MIT
# Cocotb testbench for npu_top (hw/rtl/npu/npu_top.sv).

from __future__ import annotations

import os
import struct
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from q80 import QTensor, gemv_q80, quantize_q80  # noqa: E402

# Register map (matches hw/rtl/npu/npu_pkg.sv)
REG_CTRL           = 0x00
REG_STATUS         = 0x04
REG_DIM_D          = 0x08
REG_DIM_N          = 0x0C
REG_GROUP_SIZE     = 0x10
REG_TILE_ROWS      = 0x14
REG_PERF_CYCLES    = 0x20
REG_PERF_COMPUTE   = 0x24
REG_PERF_STALL_IN  = 0x28
REG_PERF_STALL_OUT = 0x2C
REG_PERF_BYTES_IN  = 0x30
REG_PERF_BYTES_OUT = 0x34
REG_ID             = 0x38

NPU_MAGIC_ID       = 0x4E505531

CMD_LOAD_X         = 1
CMD_LOAD_XS        = 2
CMD_RUN_GEMV       = 3
CMD_CLEAR_PERF     = 4

OUT_GROUP_PSUM     = 0
OUT_ROW_ACC        = 1
OUT_ROW_FP32       = 2


def pack_weights(
    w_q: np.ndarray,
    gs: int,
    tile_rows: int = 1,
    n_lanes: int = 8,
    w_s: np.ndarray | None = None,
) -> bytes:
    """Pack weight matrix (and optional group scales) for NPU streaming."""
    d, n = w_q.shape
    eff_lanes = min(gs, n_lanes)
    num_groups = n // gs
    steps = gs // eff_lanes
    if tile_rows == 1 and w_s is None:
        raw = w_q.astype(np.int8).tobytes()
    else:
        chunks = []
        for rb in range(0, d, tile_rows):
            ra = min(tile_rows, d - rb)
            for g in range(num_groups):
                for s in range(steps):
                    c0 = g * gs + s * eff_lanes
                    for r in range(ra):
                        chunks.append(w_q[rb + r, c0: c0 + eff_lanes].astype(np.int8).tobytes())
                if w_s is not None:
                    for r in range(ra):
                        chunks.append(struct.pack("<f", float(w_s[rb + r, g])))
        raw = b"".join(chunks)
    if len(raw) % 8 != 0:
        raw += b"\x00" * (8 - (len(raw) % 8))
    return raw


def unpack_psum_tiles(words: np.ndarray, d: int, num_groups: int, tile_rows: int) -> np.ndarray:
    """Reshape flat emitted psum words (tile, group, r_in_tile) -> (d, num_groups)."""
    if tile_rows == 1:
        return words.reshape(d, num_groups)
    out = np.zeros((d, num_groups), dtype=words.dtype)
    idx = 0
    for rb in range(0, d, tile_rows):
        ra = min(tile_rows, d - rb)
        tile = words[idx: idx + ra * num_groups].reshape(num_groups, ra)
        out[rb: rb + ra, :] = tile.T
        idx += ra * num_groups
    return out


class NpuDriver:
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

    async def start_cmd(self, cmd: int, out_mode: int = OUT_GROUP_PSUM):
        ctrl = 1 | ((cmd & 0xF) << 2) | ((out_mode & 0x3) << 6)
        await self.write_reg(REG_CTRL, ctrl)

    async def send_stream(self, payload: bytes, stall_prob: float = 0.0, rng=None):
        d = self.dut
        if len(payload) % 8 != 0:
            payload = payload + b"\x00" * (8 - (len(payload) % 8))
        n_beats = len(payload) // 8
        for i in range(n_beats):
            while stall_prob > 0.0 and rng.random() < stall_prob:
                d.s_axis_tvalid.value = 0
                await RisingEdge(d.clk)
            word = int.from_bytes(payload[i * 8: (i + 1) * 8], "little")
            d.s_axis_tdata.value = word
            d.s_axis_tvalid.value = 1
            d.s_axis_tlast.value = 1 if (i == n_beats - 1) else 0
            while True:
                await RisingEdge(d.clk)
                if d.s_axis_tready.value:
                    break
        d.s_axis_tvalid.value = 0
        d.s_axis_tlast.value = 0

    async def recv_stream_words(self, expected_words: int, stall_prob: float = 0.0, rng=None) -> list[int]:
        d = self.dut
        words: list[int] = []
        while len(words) < expected_words:
            if stall_prob > 0.0 and rng.random() < stall_prob:
                d.m_axis_tready.value = 0
                await RisingEdge(d.clk)
                continue
            d.m_axis_tready.value = 1
            await RisingEdge(d.clk)
            if d.m_axis_tvalid.value and d.m_axis_tready.value:
                raw64 = int(d.m_axis_tdata.value)
                keep = int(d.m_axis_tkeep.value)
                w0 = raw64 & 0xFFFFFFFF
                w1 = (raw64 >> 32) & 0xFFFFFFFF
                words.append(w0)
                if keep == 0xFF:
                    words.append(w1)
                if d.m_axis_tlast.value:
                    break
        d.m_axis_tready.value = 1
        return words

    async def wait_done(self, timeout_cycles: int = 200000):
        for _ in range(timeout_cycles):
            st = await self.read_reg(REG_STATUS)
            if st & 0x2:
                return
        raise TimeoutError("NPU did not assert status_done within timeout")

    async def load_x(self, x_q: np.ndarray):
        n = x_q.size
        await self.write_reg(REG_DIM_N, n)
        await self.start_cmd(CMD_LOAD_X)
        await self.send_stream(x_q.astype(np.int8).tobytes())
        await self.wait_done()

    async def load_xs(self, x_s: np.ndarray, n: int, gs: int):
        await self.write_reg(REG_DIM_N, n)
        await self.write_reg(REG_GROUP_SIZE, gs)
        await self.start_cmd(CMD_LOAD_XS)
        await self.send_stream(x_s.astype(np.float32).tobytes())
        await self.wait_done()

    async def run_gemv(
        self,
        w_q: np.ndarray,
        gs: int,
        tile_rows: int = 1,
        out_mode: int = OUT_GROUP_PSUM,
        w_s: np.ndarray | None = None,
        stall_prob: float = 0.0,
        rng=None,
    ) -> np.ndarray:
        d, n = w_q.shape
        num_groups = n // gs
        expected_words = d * num_groups if out_mode == OUT_GROUP_PSUM else d

        await self.write_reg(REG_DIM_D, d)
        await self.write_reg(REG_DIM_N, n)
        await self.write_reg(REG_GROUP_SIZE, gs)
        await self.write_reg(REG_TILE_ROWS, tile_rows)
        await self.start_cmd(CMD_RUN_GEMV, out_mode=out_mode)

        payload = pack_weights(w_q, gs, tile_rows=tile_rows, w_s=w_s)
        sender = cocotb.start_soon(self.send_stream(payload, stall_prob=stall_prob, rng=rng))
        words = await self.recv_stream_words(expected_words, stall_prob=stall_prob, rng=rng)
        await sender
        await self.wait_done()

        raw_u32 = np.array(words, dtype=np.uint32)
        if out_mode == OUT_GROUP_PSUM:
            return unpack_psum_tiles(raw_u32.view(np.int32), d, num_groups, tile_rows)
        if out_mode == OUT_ROW_ACC:
            return raw_u32.view(np.int32)
        return raw_u32.view(np.float32)


@cocotb.test()
async def test_registers_and_id(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = NpuDriver(dut)
    await drv.reset()

    magic = await drv.read_reg(REG_ID)
    assert magic == NPU_MAGIC_ID, f"bad magic {magic:#x}"

    st = await drv.read_reg(REG_STATUS)
    n_rows = (st >> 8) & 0xFF
    n_lanes = (st >> 16) & 0xFF
    assert (n_rows, n_lanes) == (8, 8), f"unexpected geometry {(n_rows, n_lanes)}"

    await drv.write_reg(REG_DIM_D, 288)
    await drv.write_reg(REG_DIM_N, 768)
    await drv.write_reg(REG_GROUP_SIZE, 32)
    assert await drv.read_reg(REG_DIM_D) == 288
    assert await drv.read_reg(REG_DIM_N) == 768
    assert await drv.read_reg(REG_GROUP_SIZE) == 32


@cocotb.test()
async def test_gemv_untiled_and_tiled_modes(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = NpuDriver(dut)
    await drv.reset()
    rng = np.random.default_rng(42)

    # Test 1: Untiled (tile_rows=1), GS=32
    d, n, gs = 16, 64, 32
    wq = rng.integers(-127, 128, size=(d, n), dtype=np.int8)
    xq = rng.integers(-127, 128, size=(1, n), dtype=np.int8)
    ws = rng.uniform(0.001, 0.05, size=(d, n // gs)).astype(np.float32)
    xs = rng.uniform(0.001, 0.05, size=(1, n // gs)).astype(np.float32)
    ref_psum, ref_fp32 = gemv_q80(QTensor(wq, ws), QTensor(xq, xs), gs)

    await drv.load_x(xq)
    got_psum = await drv.run_gemv(wq, gs, tile_rows=1, out_mode=OUT_GROUP_PSUM)
    assert np.array_equal(got_psum, ref_psum), f"mismatch: {got_psum} vs {ref_psum}"

    # Test 2: Tiled 8x8 with tail rows (d=20 = 8 + 8 + 4, n=64, GS=32) + x reuse!
    d2 = 20
    wq2 = rng.integers(-127, 128, size=(d2, n), dtype=np.int8)
    ws2 = rng.uniform(0.001, 0.05, size=(d2, n // gs)).astype(np.float32)
    ref_psum2, _ = gemv_q80(QTensor(wq2, ws2), QTensor(xq, xs), gs)
    got_psum2 = await drv.run_gemv(wq2, gs, tile_rows=8, out_mode=OUT_GROUP_PSUM)
    assert np.array_equal(got_psum2, ref_psum2)

    # Test 3: OUT_ROW_ACC (integer row sum across groups)
    got_row_acc = await drv.run_gemv(wq2, gs, tile_rows=8, out_mode=OUT_ROW_ACC)
    assert np.array_equal(got_row_acc, ref_psum2.sum(axis=1))

    # Test 4: OUT_ROW_FP32 (on-chip Q8_0 dequant + IEEE-754 FP32 row accumulation)
    await drv.load_xs(xs, n, gs)
    got_fp32 = await drv.run_gemv(wq, gs, tile_rows=8, out_mode=OUT_ROW_FP32, w_s=ws)
    rel_err = np.max(np.abs(got_fp32 - ref_fp32)) / (np.max(np.abs(ref_fp32)) + 1e-12)
    assert rel_err < 1e-6, f"FP32 max rel err {rel_err}: {got_fp32} vs {ref_fp32}"


@cocotb.test()
async def test_real_model_vectors_and_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    drv = NpuDriver(dut)
    await drv.reset()
    rng = np.random.default_rng(7)

    # Verify against real GEMV vectors from stories260K (GS=4, includes d=172 and n=172)
    # and stories15M (GS=32, d=288, n=288).
    vec_root = os.path.join(ROOT, "tests", "vectors")
    cases = [
        ("260K", "wq_L0_p0.npz", 4, 8),
        ("260K", "w1_L0_p0.npz", 4, 8),   # d=172, n=64, GS=4 (tail rows + sub-lane GS)
        ("260K", "w2_L0_p0.npz", 4, 1),   # d=64, n=172, GS=4 (untiled odd-64b row length)
        ("15M",  "wq_L0_p0.npz", 32, 8),  # d=288, n=288, GS=32
    ]

    for subdir, fname, gs, tile_rows in cases:
        path = os.path.join(vec_root, subdir, fname)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        wq, ws = data["w"], data["ws"]
        xq, xs = data["xq"], data["xs"]
        ref_psum, ref_out = data["psum"], data["out"]

        await drv.start_cmd(CMD_CLEAR_PERF)
        await drv.load_x(xq)
        got_psum = await drv.run_gemv(wq, gs, tile_rows=tile_rows, out_mode=OUT_GROUP_PSUM)
        assert np.array_equal(got_psum, ref_psum), f"{subdir}/{fname} psum mismatch"

        # Also test on-chip FP32 mode on the same real weights/activations
        await drv.load_xs(xs, wq.shape[1], gs)
        got_fp32 = await drv.run_gemv(wq, gs, tile_rows=tile_rows, out_mode=OUT_ROW_FP32, w_s=ws)
        assert np.allclose(got_fp32, ref_out, rtol=1e-5, atol=1e-5), f"{subdir}/{fname} fp32 mismatch"

    # Stress backpressure on both s_axis and m_axis and verify stall counters
    await drv.start_cmd(CMD_CLEAR_PERF)
    wq = rng.integers(-127, 128, size=(16, 64), dtype=np.int8)
    xq = rng.integers(-127, 128, size=(1, 64), dtype=np.int8)
    ws = np.ones((16, 2), dtype=np.float32)
    xs = np.ones((1, 2), dtype=np.float32)
    ref_psum, _ = gemv_q80(QTensor(wq, ws), QTensor(xq, xs), 32)

    await drv.load_x(xq)
    got_psum = await drv.run_gemv(wq, 32, tile_rows=8, out_mode=OUT_GROUP_PSUM, stall_prob=0.35, rng=rng)
    assert np.array_equal(got_psum, ref_psum)

    cyc = await drv.read_reg(REG_PERF_CYCLES)
    comp = await drv.read_reg(REG_PERF_COMPUTE)
    st_in = await drv.read_reg(REG_PERF_STALL_IN)
    st_out = await drv.read_reg(REG_PERF_STALL_OUT)
    b_in = await drv.read_reg(REG_PERF_BYTES_IN)
    b_out = await drv.read_reg(REG_PERF_BYTES_OUT)
    dut._log.info(f"Perf counters: cycles={cyc} compute={comp} stall_in={st_in} stall_out={st_out} "
                  f"bytes_in={b_in} bytes_out={b_out}")
    assert comp == (16 // 8) * (64 // 8)  # 2 tiles * 8 steps = 16 MAC array cycles
    assert b_in == 64 + 16 * 64           # 64B x + 1024B weights
    assert b_out == 16 * 2 * 4            # 32 int32 partial sums = 128B
    assert st_in > 0 and st_out > 0
