/* SPDX-License-Identifier: MIT
 * accel_rtl.cpp - Verilated SystemVerilog RTL backend for NPU and Compute GPU.
 *
 * Instantiates Vnpu_top (hw/rtl/npu/npu_top.sv) and Vgpu_top (hw/rtl/gpu/gpu_top.sv)
 * directly in-process and clocks them cycle-by-cycle via AXI4-Lite and 64-bit
 * AXI4-Stream interfaces.
 */
#include "accel_backend.h"
#include "accel_hal.h"

#include "Vnpu_top.h"
#include "Vgpu_top.h"
#include <verilated.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

template <typename Top>
struct RtlDevice {
    VerilatedContext *ctx = nullptr;
    Top              *top = nullptr;
    uint64_t          sim_cycles = 0;

    void init() {
        ctx = new VerilatedContext;
        top = new Top(ctx);
        top->clk           = 0;
        top->rst_n         = 0;
        top->s_axi_awaddr  = 0;
        top->s_axi_awvalid = 0;
        top->s_axi_wdata   = 0;
        top->s_axi_wstrb   = 0xF;
        top->s_axi_wvalid  = 0;
        top->s_axi_bready  = 0;
        top->s_axi_araddr  = 0;
        top->s_axi_arvalid = 0;
        top->s_axi_rready  = 0;
        top->s_axis_tdata  = 0;
        top->s_axis_tvalid = 0;
        top->s_axis_tlast  = 0;
        top->m_axis_tready = 0;
        top->eval();

        for (int i = 0; i < 5; i++) tick();
        top->rst_n = 1;
        for (int i = 0; i < 2; i++) tick();
        sim_cycles = 0;
    }

    void close() {
        if (top) {
            top->final();
            delete top;
            top = nullptr;
        }
        if (ctx) {
            delete ctx;
            ctx = nullptr;
        }
    }

    inline void tick() {
        top->eval();
        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();
        sim_cycles++;
    }

    void reg_write(uint32_t off, uint32_t val) {
        top->s_axi_awaddr  = off & 0x7Fu;
        top->s_axi_awvalid = 1;
        top->s_axi_wdata   = val;
        top->s_axi_wstrb   = 0xF;
        top->s_axi_wvalid  = 1;
        top->s_axi_bready  = 1;
        top->eval();

        bool aw_done = false, w_done = false;
        while (!(aw_done && w_done)) {
            bool aw_hs = top->s_axi_awvalid && top->s_axi_awready;
            bool w_hs  = top->s_axi_wvalid  && top->s_axi_wready;
            tick();
            if (aw_hs) { top->s_axi_awvalid = 0; aw_done = true; }
            if (w_hs)  { top->s_axi_wvalid  = 0; w_done  = true; }
            top->eval();
        }
        while (!top->s_axi_bvalid) {
            tick();
        }
        tick();
        top->s_axi_bready = 0;
        top->eval();
    }

    uint32_t reg_read(uint32_t off) {
        top->s_axi_araddr  = off & 0x7Fu;
        top->s_axi_arvalid = 1;
        top->s_axi_rready  = 1;
        top->eval();

        while (!(top->s_axi_arvalid && top->s_axi_arready)) {
            tick();
        }
        tick();
        top->s_axi_arvalid = 0;
        top->eval();

        while (!top->s_axi_rvalid) {
            tick();
        }
        uint32_t val = top->s_axi_rdata;
        tick();
        top->s_axi_rready = 0;
        top->eval();
        return val;
    }

    void stream(const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes) {
        const uint8_t *tx_ptr = static_cast<const uint8_t *>(tx);
        uint8_t       *rx_ptr = static_cast<uint8_t *>(rx);

        size_t tx_beats = (tx_bytes + 7u) >> 3;
        size_t rx_beats = (rx_bytes + 7u) >> 3;
        size_t tx_idx   = 0;
        size_t rx_idx   = 0;

        while (tx_idx < tx_beats || rx_idx < rx_beats || !top->irq) {
            if (tx_idx < tx_beats) {
                uint64_t beat = 0;
                size_t base = tx_idx << 3;
                size_t rem  = tx_bytes - base;
                size_t copy = (rem < 8u) ? rem : 8u;
                std::memcpy(&beat, tx_ptr + base, copy);
                top->s_axis_tdata  = beat;
                top->s_axis_tvalid = 1;
                top->s_axis_tlast  = (tx_idx + 1u == tx_beats) ? 1 : 0;
            } else {
                top->s_axis_tdata  = 0;
                top->s_axis_tvalid = 0;
                top->s_axis_tlast  = 0;
            }

            top->m_axis_tready = (rx_idx < rx_beats) ? 1 : 0;
            top->eval();

            bool     tx_fire = top->s_axis_tvalid && top->s_axis_tready;
            bool     rx_fire = top->m_axis_tvalid && top->m_axis_tready;
            uint64_t rx_beat = top->m_axis_tdata;

            tick();

            if (tx_fire) {
                tx_idx++;
            }
            if (rx_fire) {
                size_t base = rx_idx << 3;
                size_t rem  = rx_bytes - base;
                size_t copy = (rem < 8u) ? rem : 8u;
                std::memcpy(rx_ptr + base, &rx_beat, copy);
                rx_idx++;
            }
        }

        top->s_axis_tvalid = 0;
        top->s_axis_tlast  = 0;
        top->m_axis_tready = 0;
        top->eval();
    }
};

static RtlDevice<Vnpu_top> g_npu;
static RtlDevice<Vgpu_top> g_gpu;

static int rtl_init(void) {
    g_npu.init();
    g_gpu.init();
    return 0;
}

static void rtl_close(void) {
    g_npu.close();
    g_gpu.close();
}

static void rtl_reg_write(AccelDev dev, uint32_t off, uint32_t val) {
    if (dev == DEV_NPU) g_npu.reg_write(off, val);
    else                g_gpu.reg_write(off, val);
}

static uint32_t rtl_reg_read(AccelDev dev, uint32_t off) {
    if (dev == DEV_NPU) return g_npu.reg_read(off);
    else                return g_gpu.reg_read(off);
}

static void rtl_stream(AccelDev dev, const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes) {
    if (dev == DEV_NPU) g_npu.stream(tx, tx_bytes, rx, rx_bytes);
    else                g_gpu.stream(tx, tx_bytes, rx, rx_bytes);
}

static void *rtl_alloc(size_t bytes) {
    return std::malloc(bytes);
}

static void rtl_free(void *p) {
    std::free(p);
}

} // namespace

extern "C" const AccelBackend accel_backend_rtl = {
    "rtl",
    rtl_init,
    rtl_close,
    rtl_reg_write,
    rtl_reg_read,
    rtl_stream,
    rtl_alloc,
    rtl_free
};
