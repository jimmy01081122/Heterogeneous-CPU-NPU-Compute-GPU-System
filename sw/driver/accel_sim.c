/* SPDX-License-Identifier: MIT
 * accel_sim.c - Cycle-accurate C behavioral & ISA simulator backend.
 *
 * Models:
 *   1. NPU (npu_top.sv): AXI4-Lite registers, 1 KB activation buffer,
 *      8x8 INT8 MAC array, tiled & untiled weight streams, and hardware
 *      performance counters (NPU_REG_PERF_CYCLES, etc.).
 *   2. Compute GPU (gpu_top.sv): AXI4-Lite registers, 256-word IMEM,
 *      4096-word (16 KB) Scratchpad Memory, 4 SIMD/SIMT lanes, 16x4 vector
 *      register file, hardware strip-mining loop (VRESET/VSTEP) with
 *      tail predicate masking, and the exact IEEE-754 / fast-math ALU
 *      approximations from hw/rtl/common/fp32_alu.sv.
 *
 * Works on both x86_64 host and 32-bit ARM (qemu-arm) without requiring C++
 * or Verilator at runtime.
 */
#include "accel_backend.h"
#include "accel_hal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Bitcast helpers ---------------------------------------------------- */
static inline float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

/* ---- Hardware-accurate FP32 special functions (from fp32_alu.sv) -------- */
static uint32_t hw_fp32_recip(uint32_t x) {
    if (((x >> 23) & 0xFFu) == 0) return (x & 0x80000000u) | 0x7F800000u;
    uint32_t sign = x & 0x80000000u;
    uint32_t mag  = x & 0x7FFFFFFFu;
    uint32_t y0   = sign | ((0x7EEF1AA0u - mag) & 0x7FFFFFFFu);
    float    fx   = u2f(mag);
    float    fy0  = u2f(y0);
    float    y1   = fy0 * (2.0f - fx * fabsf(fy0));
    float    y2   = y1  * (2.0f - fx * fabsf(y1));
    return f2u(y2);
}

static uint32_t hw_fp32_rsqrt(uint32_t x) {
    if ((x & 0x80000000u) || ((x >> 23) & 0xFFu) == 0) return 0x7FC00000u;
    uint32_t xhalf_u = ((x >> 23) - 1u) << 23 | (x & 0x007FFFFFu);
    float    xhalf   = u2f(xhalf_u);
    uint32_t yu      = 0x5F375A86u - (x >> 1);
    float    y       = u2f(yu);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return f2u(y);
}

static uint32_t hw_fp32_exp(uint32_t xu) {
    float x = u2f(xu) * u2f(0x3FB8AA3Bu); /* x * log2(e) */
    if (x <= -126.0f) return 0u;
    if (x >= 127.0f)  return 0x7F800000u;
    int32_t n_int = (int32_t)lrintf(x);
    float   f     = x - (float)n_int;
    if (f < 0.0f) {
        n_int -= 1;
        f      = x - (float)n_int;
    }
    const float c1 = u2f(0x3F31F519u);
    const float c2 = u2f(0x3E69089Au);
    const float c3 = u2f(0x3D9E4B18u);
    float p = 1.0f + f * (c1 + f * (c2 + f * c3));
    uint32_t pu = f2u(p);
    int exp_out = (int)((pu >> 23) & 0xFFu) + n_int;
    if (exp_out <= 0)   return 0u;
    if (exp_out >= 255) return 0x7F800000u;
    return ((uint32_t)exp_out << 23) | (pu & 0x007FFFFFu);
}

static uint32_t hw_fp32_silu(uint32_t xu) {
    uint32_t neg_x  = xu ^ 0x80000000u;
    uint32_t exp_nx = hw_fp32_exp(neg_x);
    float    denom  = 1.0f + u2f(exp_nx);
    uint32_t inv    = hw_fp32_recip(f2u(denom));
    return f2u(u2f(xu) * u2f(inv));
}

/* ---- Simulated NPU State ------------------------------------------------ */
static struct {
    uint32_t dim_d;
    uint32_t dim_n;
    uint32_t gs;
    uint32_t tile_rows;
    uint32_t cmd;
    uint32_t out_mode;
    uint32_t status;
    uint32_t perf_cycles;
    uint32_t perf_compute;
    uint32_t perf_bytes_in;
    uint32_t perf_bytes_out;
    int8_t   act_x[1024];
    float    act_xs[256];
} s_npu;

/* ---- Simulated Compute GPU State ---------------------------------------- */
#define SIM_GPU_LANES 4
#define SIM_SPM_WORDS 4096
#define SIM_IMEM_WORDS 256

static struct {
    uint32_t vec_len;
    uint32_t dma_addr;
    uint32_t dma_len;
    uint32_t imem_addr;
    uint32_t param0;
    uint32_t param1;
    uint32_t start_pc;
    uint32_t cmd;
    uint32_t status;
    uint32_t perf_cycles;
    uint32_t perf_insts;
    uint32_t perf_bytes_in;
    uint32_t perf_bytes_out;
    uint32_t imem[SIM_IMEM_WORDS];
    uint32_t spm[SIM_SPM_WORDS];
    uint32_t vrf[16][SIM_GPU_LANES];
} s_gpu;

static int sim_init(void) {
    memset(&s_npu, 0, sizeof(s_npu));
    memset(&s_gpu, 0, sizeof(s_gpu));
    s_npu.gs        = 32;
    s_npu.tile_rows = 8;
    s_gpu.vec_len   = 4;
    return 0;
}

static void sim_close(void) {}

static void sim_gpu_exec_kernel(void) {
    uint32_t pc        = s_gpu.start_pc & (SIM_IMEM_WORDS - 1u);
    uint32_t vloop_idx = 0;
    uint32_t guard     = 0;

    memset(s_gpu.vrf, 0, sizeof(s_gpu.vrf));
    s_gpu.perf_cycles += 1; /* S_FETCH initial */

    while (guard++ < 1000000u) {
        uint32_t inst   = s_gpu.imem[pc & (SIM_IMEM_WORDS - 1u)];
        uint32_t opcode = (inst >> 26) & 0x3Fu;
        uint32_t rd     = (inst >> 22) & 0xFu;
        uint32_t rs1    = (inst >> 18) & 0xFu;
        uint32_t rs2    = (inst >> 14) & 0xFu;
        uint32_t imm14  = inst & 0x3FFFu;
        int32_t  imm_s  = (imm14 & 0x2000u) ? (int32_t)(imm14 | ~0x3FFFu) : (int32_t)imm14;

        int lane_active[SIM_GPU_LANES];
        for (int l = 0; l < SIM_GPU_LANES; l++) {
            lane_active[l] = (vloop_idx + (uint32_t)l < s_gpu.vec_len) ? 1 : 0;
        }

        /* Single-cycle execution pipeline in gpu_controller.sv */
        s_gpu.perf_cycles += 1;
        s_gpu.perf_insts  += 1;

        if (opcode == 0x3Fu) { /* HALT */
            break;
        } else if (opcode == 0x38u) { /* VRESET */
            vloop_idx = 0;
            pc++;
        } else if (opcode == 0x39u) { /* VSTEP */
            if (vloop_idx + SIM_GPU_LANES < s_gpu.vec_len) {
                vloop_idx += SIM_GPU_LANES;
                pc = imm14 & (SIM_IMEM_WORDS - 1u);
            } else {
                vloop_idx = 0;
                pc++;
            }
        } else if (opcode == 0x3Eu) { /* JMP */
            pc = imm14 & (SIM_IMEM_WORDS - 1u);
        } else if (opcode == 0x01u) { /* VLD */
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) {
                    uint32_t addr = (imm14 + vloop_idx + (uint32_t)l) & (SIM_SPM_WORDS - 1u);
                    s_gpu.vrf[rd][l] = s_gpu.spm[addr];
                }
            }
            pc++;
        } else if (opcode == 0x02u) { /* VST: spm[imm14 + vloop_idx + l] <= rs2[l] */
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) {
                    uint32_t addr = (imm14 + vloop_idx + (uint32_t)l) & (SIM_SPM_WORDS - 1u);
                    s_gpu.spm[addr] = s_gpu.vrf[rs2][l];
                }
            }
            pc++;
        } else if (opcode == 0x03u) { /* SLD */
            uint32_t val = s_gpu.spm[imm14 & (SIM_SPM_WORDS - 1u)];
            for (int l = 0; l < SIM_GPU_LANES; l++) s_gpu.vrf[rd][l] = val;
            pc++;
        } else if (opcode == 0x04u) { /* SST: spm[imm14] <= rs2[0] */
            s_gpu.spm[imm14 & (SIM_SPM_WORDS - 1u)] = s_gpu.vrf[rs2][0];
            pc++;
        } else if (opcode == 0x05u) { /* VLD_ABS */
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) {
                    uint32_t addr = (imm14 + s_gpu.vrf[rs1][l]) & (SIM_SPM_WORDS - 1u);
                    s_gpu.vrf[rd][l] = s_gpu.spm[addr];
                }
            }
            pc++;
        } else if (opcode == 0x06u) { /* VST_ABS: spm[imm14 + rs1[l]] <= rs2[l] */
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) {
                    uint32_t addr = (imm14 + s_gpu.vrf[rs1][l]) & (SIM_SPM_WORDS - 1u);
                    s_gpu.spm[addr] = s_gpu.vrf[rs2][l];
                }
            }
            pc++;
        } else if (opcode == 0x30u) { /* RADD: rd <= rd[0] + sum(rs1[l]) */
            float sum = u2f(s_gpu.vrf[rd][0]);
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) sum += u2f(s_gpu.vrf[rs1][l]);
            }
            uint32_t res = f2u(sum);
            for (int l = 0; l < SIM_GPU_LANES; l++) s_gpu.vrf[rd][l] = res;
            pc++;
        } else if (opcode == 0x31u) { /* RMAX: rd <= max(rd[0], rs1[l]) */
            float mx = u2f(s_gpu.vrf[rd][0]);
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                if (lane_active[l]) {
                    float v = u2f(s_gpu.vrf[rs1][l]);
                    if (v > mx) mx = v;
                }
            }
            uint32_t res = f2u(mx);
            for (int l = 0; l < SIM_GPU_LANES; l++) s_gpu.vrf[rd][l] = res;
            pc++;
        } else {
            /* Per-lane ALU / Immediate ops */
            int is_bcast = (opcode == 0x08u || opcode == 0x09u);
            uint32_t out[SIM_GPU_LANES];
            for (int l = 0; l < SIM_GPU_LANES; l++) {
                uint32_t a = s_gpu.vrf[rs1][l];
                uint32_t b = s_gpu.vrf[rs2][l];
                uint32_t r = 0;
                switch (opcode) {
                    case 0x08u: r = (uint32_t)imm_s; break; /* LDI */
                    case 0x09u: r = (imm14 & 1u) ? s_gpu.param1 : s_gpu.param0; break; /* LD_PARAM */
                    case 0x0Au: r = vloop_idx + (uint32_t)l; break; /* LD_LANE */
                    case 0x0Bu: r = a; break; /* MOV */
                    case 0x10u: r = a + b; break; /* IADD */
                    case 0x11u: r = a - b; break; /* ISUB */
                    case 0x12u: r = (uint32_t)((int32_t)a * (int32_t)b); break; /* IMUL */
                    case 0x13u: r = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 16); break;
                    case 0x14u: r = a & b; break;
                    case 0x15u: r = a | b; break;
                    case 0x16u: r = a ^ b; break;
                    case 0x17u: r = a << (b & 31u); break; /* SHL */
                    case 0x18u: r = (uint32_t)((int32_t)a >> (b & 31u)); break; /* SRA */
                    case 0x20u: r = f2u(u2f(a) + u2f(b)); break; /* FADD */
                    case 0x21u: r = f2u(u2f(a) - u2f(b)); break; /* FSUB */
                    case 0x22u: r = f2u(u2f(a) * u2f(b)); break; /* FMUL */
                    case 0x23u: r = f2u(u2f(a) > u2f(b) ? u2f(a) : u2f(b)); break; /* FMAX */
                    case 0x24u: r = f2u(u2f(a) < u2f(b) ? u2f(a) : u2f(b)); break; /* FMIN */
                    case 0x25u: r = f2u((float)(int32_t)a); break; /* I2F */
                    case 0x26u: r = (uint32_t)(int32_t)lrintf(u2f(a)); break; /* F2I */
                    case 0x28u: r = hw_fp32_recip(a); break; /* FRECIP */
                    case 0x29u: r = hw_fp32_rsqrt(a); break; /* FRSQRT */
                    case 0x2Au: r = hw_fp32_exp(a);   break; /* FEXP */
                    case 0x2Bu: r = hw_fp32_silu(a);  break; /* FSILU */
                    default:    r = 0; break;
                }
                out[l] = r;
            }
            if (opcode != 0x00u) {
                for (int l = 0; l < SIM_GPU_LANES; l++) {
                    if (is_bcast || lane_active[l]) s_gpu.vrf[rd][l] = out[l];
                }
            }
            pc++;
        }
    }
    s_gpu.status = 0x2u; /* done */
}

static void sim_reg_write(AccelDev dev, uint32_t off, uint32_t val) {
    if (dev == DEV_NPU) {
        switch (off) {
            case NPU_REG_CTRL:
                s_npu.cmd      = (val >> 2) & 0xFu;
                s_npu.out_mode = (val >> 6) & 0x3u;
                if ((val & 0x2u) || s_npu.cmd == NPU_CMD_CLEAR_PERF) {
                    s_npu.perf_cycles    = 0;
                    s_npu.perf_compute   = 0;
                    s_npu.perf_bytes_in  = 0;
                    s_npu.perf_bytes_out = 0;
                    s_npu.status         = 0x2u;
                }
                break;
            case NPU_REG_DIM_D:      s_npu.dim_d     = val & 0xFFFFu; break;
            case NPU_REG_DIM_N:      s_npu.dim_n     = val & 0xFFFFu; break;
            case NPU_REG_GROUP_SIZE: s_npu.gs        = val & 0xFFFFu; break;
            case NPU_REG_TILE_ROWS:  s_npu.tile_rows = val & 0xFFu;   break;
            default: break;
        }
    } else {
        switch (off) {
            case GPU_REG_CTRL:
                s_gpu.cmd = (val >> 2) & 0xFu;
                if ((val & 0x2u) || s_gpu.cmd == GPU_CMD_CLEAR_PERF) {
                    s_gpu.perf_cycles    = 0;
                    s_gpu.perf_insts     = 0;
                    s_gpu.perf_bytes_in  = 0;
                    s_gpu.perf_bytes_out = 0;
                    s_gpu.status         = 0x2u;
                }
                break;
            case GPU_REG_VEC_LEN:   s_gpu.vec_len   = val; break;
            case GPU_REG_DMA_ADDR:  s_gpu.dma_addr  = val & (SIM_SPM_WORDS - 1u); break;
            case GPU_REG_DMA_LEN:   s_gpu.dma_len   = val; break;
            case GPU_REG_IMEM_ADDR: s_gpu.imem_addr = val & (SIM_IMEM_WORDS - 1u); break;
            case GPU_REG_IMEM_DATA:
                s_gpu.imem[s_gpu.imem_addr & (SIM_IMEM_WORDS - 1u)] = val;
                s_gpu.imem_addr = (s_gpu.imem_addr + 1u) & (SIM_IMEM_WORDS - 1u);
                break;
            case GPU_REG_PARAM0:    s_gpu.param0    = val; break;
            case GPU_REG_PARAM1:    s_gpu.param1    = val; break;
            case GPU_REG_START_PC:  s_gpu.start_pc  = val & (SIM_IMEM_WORDS - 1u); break;
            default: break;
        }
    }
}

static uint32_t sim_reg_read(AccelDev dev, uint32_t off) {
    if (dev == DEV_NPU) {
        switch (off) {
            case NPU_REG_STATUS:         return s_npu.status;
            case NPU_REG_DIM_D:          return s_npu.dim_d;
            case NPU_REG_DIM_N:          return s_npu.dim_n;
            case NPU_REG_GROUP_SIZE:     return s_npu.gs;
            case NPU_REG_TILE_ROWS:      return s_npu.tile_rows;
            case NPU_REG_PERF_CYCLES:    return s_npu.perf_cycles;
            case NPU_REG_PERF_COMPUTE:   return s_npu.perf_compute;
            case NPU_REG_PERF_BYTES_IN:  return s_npu.perf_bytes_in;
            case NPU_REG_PERF_BYTES_OUT: return s_npu.perf_bytes_out;
            case NPU_REG_ID:             return NPU_MAGIC_ID;
            default:                     return 0u;
        }
    } else {
        switch (off) {
            case GPU_REG_STATUS:         return s_gpu.status;
            case GPU_REG_VEC_LEN:        return s_gpu.vec_len;
            case GPU_REG_DMA_ADDR:       return s_gpu.dma_addr;
            case GPU_REG_DMA_LEN:        return s_gpu.dma_len;
            case GPU_REG_PARAM0:         return s_gpu.param0;
            case GPU_REG_PARAM1:         return s_gpu.param1;
            case GPU_REG_START_PC:       return s_gpu.start_pc;
            case GPU_REG_PERF_CYCLES:    return s_gpu.perf_cycles;
            case GPU_REG_PERF_INSTS:     return s_gpu.perf_insts;
            case GPU_REG_PERF_BYTES_IN:  return s_gpu.perf_bytes_in;
            case GPU_REG_PERF_BYTES_OUT: return s_gpu.perf_bytes_out;
            case GPU_REG_ID:             return GPU_MAGIC_ID;
            default:                     return 0u;
        }
    }
}

static void sim_stream(AccelDev dev, const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes) {
    if (dev == DEV_NPU) {
        if (s_npu.cmd == NPU_CMD_LOAD_X) {
            size_t copy = tx_bytes < sizeof(s_npu.act_x) ? tx_bytes : sizeof(s_npu.act_x);
            memcpy(s_npu.act_x, tx, copy);
            uint32_t beats = (uint32_t)((tx_bytes + 7u) >> 3);
            s_npu.perf_cycles   += beats + 2u;
            s_npu.perf_bytes_in += beats * 8u;
            s_npu.status         = 0x6u; /* done | x_loaded */
        } else if (s_npu.cmd == NPU_CMD_RUN_GEMV) {
            int d         = (int)s_npu.dim_d;
            int n         = (int)s_npu.dim_n;
            int gs        = (int)s_npu.gs;
            int tile_cap  = (s_npu.tile_rows == 0 || s_npu.tile_rows > 8) ? 8 : (int)s_npu.tile_rows;
            int eff_lanes = (gs < 8) ? gs : 8;
            int ng        = n / gs;
            int steps     = gs / eff_lanes;
            const int8_t *w_ptr = (const int8_t *)tx;
            int32_t      *out_p = (int32_t *)rx;
            size_t        w_pos = 0;
            size_t        o_pos = 0;

            for (int row_base = 0; row_base < d; row_base += tile_cap) {
                int r_act = (d - row_base < tile_cap) ? (d - row_base) : tile_cap;
                for (int g = 0; g < ng; g++) {
                    int32_t psum[8] = {0};
                    for (int s = 0; s < steps; s++) {
                        int col = g * gs + s * eff_lanes;
                        for (int r = 0; r < r_act; r++) {
                            int32_t dot = 0;
                            for (int c = 0; c < eff_lanes; c++) {
                                dot += (int32_t)w_ptr[w_pos + c] * (int32_t)s_npu.act_x[col + c];
                            }
                            w_pos   += (size_t)eff_lanes;
                            psum[r] += dot;
                        }
                        s_npu.perf_compute += 1u;
                    }
                    for (int r = 0; r < r_act; r++) {
                        out_p[o_pos++] = psum[r];
                    }
                    /* Exact cycle model matching npu_top.sv FSM:
                     * r_act * steps weight cycles + 1 POST cycle + r_act EMIT cycles */
                    s_npu.perf_cycles += (uint32_t)(r_act * steps + 1 + r_act);
                }
            }
            s_npu.perf_cycles    += 2u; /* drain */
            s_npu.perf_bytes_in  += (uint32_t)(((tx_bytes + 7u) >> 3) * 8u);
            s_npu.perf_bytes_out += (uint32_t)rx_bytes;
            s_npu.status          = 0x6u;
        }
    } else {
        if (s_gpu.cmd == GPU_CMD_DMA_H2D) {
            uint32_t n_words = s_gpu.dma_len;
            uint32_t addr    = s_gpu.dma_addr & (SIM_SPM_WORDS - 1u);
            memcpy(&s_gpu.spm[addr], tx, (size_t)n_words * sizeof(uint32_t));
            s_gpu.perf_cycles   += (n_words + 1u) >> 1;
            s_gpu.perf_bytes_in += n_words * 4u;
            s_gpu.status         = 0x2u;
        } else if (s_gpu.cmd == GPU_CMD_DMA_D2H) {
            uint32_t n_words = s_gpu.dma_len;
            uint32_t addr    = s_gpu.dma_addr & (SIM_SPM_WORDS - 1u);
            memcpy(rx, &s_gpu.spm[addr], (size_t)n_words * sizeof(uint32_t));
            s_gpu.perf_cycles    += (n_words + 1u) >> 1;
            s_gpu.perf_bytes_out += n_words * 4u;
            s_gpu.status          = 0x2u;
        } else if (s_gpu.cmd == GPU_CMD_RUN_KERNEL) {
            sim_gpu_exec_kernel();
        }
    }
}

static void *sim_alloc(size_t bytes) {
    return malloc(bytes);
}

static void sim_free(void *p) {
    free(p);
}

const AccelBackend accel_backend_sim = {
    "sim",
    sim_init,
    sim_close,
    sim_reg_write,
    sim_reg_read,
    sim_stream,
    sim_alloc,
    sim_free
};
