/* SPDX-License-Identifier: MIT
 * accel.c - shared high-level NPU and Compute GPU driver implementation.
 *
 * Uses the AccelBackend vtable (accel_backend.h) so the exact same register
 * and stream sequences run against:
 *   - "rtl": Verilated SystemVerilog RTL (npu_top.sv + gpu_top.sv)
 *   - "sim": Cycle-accurate C behavioral / GPU ISA simulator
 *   - "hw" : PYNQ-Z2 /dev/mem + Xilinx AXI DMA + CMA physical buffers
 */
#include "accel.h"
#include "accel_backend.h"
#include "accel_hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const AccelBackend *g_be = NULL;
static AccelStats          g_stats;
static int32_t            *g_psum_scratch = NULL;
static size_t              g_psum_scratch_cap = 0;

static inline void wr(AccelDev dev, uint32_t off, uint32_t val) {
    g_stats.reg_writes[dev]++;
    g_be->reg_write(dev, off, val);
}

static inline uint32_t rd(AccelDev dev, uint32_t off) {
    g_stats.reg_reads[dev]++;
    return g_be->reg_read(dev, off);
}

static inline void xfer(AccelDev dev, const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes) {
    g_stats.calls[dev]++;
    g_stats.bytes_in[dev]  += tx_bytes;
    g_stats.bytes_out[dev] += rx_bytes;
    g_be->stream(dev, tx, tx_bytes, rx, rx_bytes);
}

int accel_init(const char *backend) {
    if (!backend || !*backend) backend = "rtl";
    memset(&g_stats, 0, sizeof(g_stats));

    if (strcmp(backend, "hw") == 0) {
        g_be = &accel_backend_hw;
    } else if (strcmp(backend, "sim") == 0) {
        g_be = &accel_backend_sim;
    } else if (strcmp(backend, "rtl") == 0) {
#ifdef HAVE_VERILATOR_RTL
        g_be = &accel_backend_rtl;
#else
        g_be = &accel_backend_sim;
#endif
    } else {
        fprintf(stderr, "[accel] unknown backend '%s' (expected 'rtl', 'sim', or 'hw')\n", backend);
        return -1;
    }

    if (g_be->init() != 0) {
        fprintf(stderr, "[accel] backend '%s' init failed\n", g_be->name);
        g_be = NULL;
        return -1;
    }

    uint32_t npu_id = rd(DEV_NPU, NPU_REG_ID);
    uint32_t gpu_id = rd(DEV_GPU, GPU_REG_ID);
    if (npu_id != NPU_MAGIC_ID || gpu_id != GPU_MAGIC_ID) {
        fprintf(stderr, "[accel] ID mismatch: NPU=0x%08x (exp 0x%08x), GPU=0x%08x (exp 0x%08x)\n",
                npu_id, NPU_MAGIC_ID, gpu_id, GPU_MAGIC_ID);
        g_be->close();
        g_be = NULL;
        return -1;
    }

    wr(DEV_NPU, NPU_REG_CTRL, npu_ctrl_word(NPU_CMD_CLEAR_PERF, 0));
    wr(DEV_GPU, GPU_REG_CTRL, gpu_ctrl_word(GPU_CMD_CLEAR_PERF));
    return 0;
}

void accel_close(void) {
    if (g_psum_scratch) {
        free(g_psum_scratch);
        g_psum_scratch = NULL;
        g_psum_scratch_cap = 0;
    }
    if (g_be) {
        g_be->close();
        g_be = NULL;
    }
}

AccelStats *accel_stats(void) {
    if (g_be) {
        g_stats.cycles[DEV_NPU] = rd(DEV_NPU, NPU_REG_PERF_CYCLES);
        g_stats.cycles[DEV_GPU] = rd(DEV_GPU, GPU_REG_PERF_CYCLES);
    }
    return &g_stats;
}

void *accel_alloc(size_t bytes) {
    size_t aligned = (bytes + 63u) & ~63u;
    void *p = g_be ? g_be->alloc(aligned) : malloc(aligned);
    if (p) memset(p, 0, aligned);
    return p;
}

void accel_free(void *p) {
    if (!p) return;
    if (g_be) g_be->free(p);
    else free(p);
}

/* ---- NPU ---------------------------------------------------------------- */

void npu_load_x(const int8_t *xq, int n) {
    wr(DEV_NPU, NPU_REG_DIM_N, (uint32_t)n);
    wr(DEV_NPU, NPU_REG_CTRL, npu_ctrl_word(NPU_CMD_LOAD_X, 0));
    xfer(DEV_NPU, xq, (size_t)n, NULL, 0);
}

size_t npu_packed_size(int d, int n) {
    size_t raw = (size_t)d * (size_t)n;
    return (raw + 7u) & ~7u;
}

size_t npu_pack_tiled(const int8_t *w, int d, int n, int gs, int tile_rows, int8_t *dst) {
    if (tile_rows <= 1) {
        size_t bytes = (size_t)d * (size_t)n;
        memcpy(dst, w, bytes);
        return bytes;
    }
    int eff_lanes = (gs < 8) ? gs : 8;
    int ng        = n / gs;
    int steps     = gs / eff_lanes;
    size_t pos    = 0;

    for (int row_base = 0; row_base < d; row_base += tile_rows) {
        int r_act = (d - row_base < tile_rows) ? (d - row_base) : tile_rows;
        for (int g = 0; g < ng; g++) {
            for (int s = 0; s < steps; s++) {
                int col = g * gs + s * eff_lanes;
                for (int r = 0; r < r_act; r++) {
                    const int8_t *src = w + (size_t)(row_base + r) * (size_t)n + col;
                    memcpy(dst + pos, src, (size_t)eff_lanes);
                    pos += (size_t)eff_lanes;
                }
            }
        }
    }
    return pos;
}

void npu_gemv_psum(const int8_t *w, int d, int n, int gs, int tile_rows, int32_t *psum) {
    int ng = n / gs;
    size_t tx_bytes = (size_t)d * (size_t)n;
    size_t rx_words = (size_t)d * (size_t)ng;
    size_t rx_bytes = rx_words * sizeof(int32_t);

    wr(DEV_NPU, NPU_REG_DIM_D,      (uint32_t)d);
    wr(DEV_NPU, NPU_REG_DIM_N,      (uint32_t)n);
    wr(DEV_NPU, NPU_REG_GROUP_SIZE, (uint32_t)gs);
    wr(DEV_NPU, NPU_REG_TILE_ROWS,  (uint32_t)tile_rows);
    wr(DEV_NPU, NPU_REG_CTRL,       npu_ctrl_word(NPU_CMD_RUN_GEMV, NPU_OUT_GROUP_PSUM));

    if (tile_rows <= 1) {
        xfer(DEV_NPU, w, tx_bytes, psum, rx_bytes);
        return;
    }

    /* When tile_rows > 1, NPU emits [tile][group][r_in_tile]; transpose to [row][group]. */
    if (rx_words > g_psum_scratch_cap) {
        free(g_psum_scratch);
        g_psum_scratch_cap = rx_words;
        g_psum_scratch = (int32_t *)malloc(rx_bytes);
    }
    xfer(DEV_NPU, w, tx_bytes, g_psum_scratch, rx_bytes);

    size_t idx = 0;
    for (int row_base = 0; row_base < d; row_base += tile_rows) {
        int r_act = (d - row_base < tile_rows) ? (d - row_base) : tile_rows;
        for (int g = 0; g < ng; g++) {
            for (int r = 0; r < r_act; r++) {
                psum[(row_base + r) * ng + g] = g_psum_scratch[idx++];
            }
        }
    }
}

/* ---- Compute GPU -------------------------------------------------------- */

void gpu_load_image(const uint32_t *words, int n) {
    for (int i = 0; i < n; i++) {
        wr(DEV_GPU, GPU_REG_IMEM_ADDR, (uint32_t)i);
        wr(DEV_GPU, GPU_REG_IMEM_DATA, words[i]);
    }
}

void gpu_write_spm(uint32_t word_addr, const void *src, int n_words) {
    if (n_words <= 0) return;
    wr(DEV_GPU, GPU_REG_DMA_ADDR, word_addr);
    wr(DEV_GPU, GPU_REG_DMA_LEN,  (uint32_t)n_words);
    wr(DEV_GPU, GPU_REG_CTRL,     gpu_ctrl_word(GPU_CMD_DMA_H2D));
    xfer(DEV_GPU, src, (size_t)n_words * sizeof(uint32_t), NULL, 0);
}

void gpu_read_spm(uint32_t word_addr, void *dst, int n_words) {
    if (n_words <= 0) return;
    wr(DEV_GPU, GPU_REG_DMA_ADDR, word_addr);
    wr(DEV_GPU, GPU_REG_DMA_LEN,  (uint32_t)n_words);
    wr(DEV_GPU, GPU_REG_CTRL,     gpu_ctrl_word(GPU_CMD_DMA_D2H));
    xfer(DEV_GPU, NULL, 0, dst, (size_t)n_words * sizeof(uint32_t));
}

void gpu_run(uint32_t start_pc, uint32_t vec_len, uint32_t param0, uint32_t param1) {
    wr(DEV_GPU, GPU_REG_START_PC, start_pc);
    wr(DEV_GPU, GPU_REG_VEC_LEN,  vec_len);
    wr(DEV_GPU, GPU_REG_PARAM0,   param0);
    wr(DEV_GPU, GPU_REG_PARAM1,   param1);
    wr(DEV_GPU, GPU_REG_CTRL,     gpu_ctrl_word(GPU_CMD_RUN_KERNEL));
    xfer(DEV_GPU, NULL, 0, NULL, 0);
}
